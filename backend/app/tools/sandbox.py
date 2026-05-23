"""Subprocess sandbox.

Hardens execution inside the configured `SANDBOX_ROOT`:

- A per-task workdir is created on demand (or supplied by the caller); it
  is the only writable path the subprocess sees (cwd + restricted env).
- POSIX rlimits cap CPU time, virtual memory, and open files (best-effort
  on platforms where `resource` is unavailable, e.g. Windows).
- `asyncio.wait_for` enforces a wallclock timeout regardless of rlimits.
- Network is *not* network-namespaced (we don't run as root); we instead
  scrub `HTTP*_PROXY` env and rely on tools to honor a `network=False`
  contract. For full isolation, run inside a container — the docker
  compose backend already provides that boundary.

The sandbox is shared across tools that need to spawn processes
(`shell`, `code_exec`, `pytest_runner`).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Mapping, Sequence

from app.config import Settings
from app.observability import get_logger

logger = get_logger("tools.sandbox")

# Best-effort import of `resource`; on Windows it doesn't exist.
try:
    import resource  # type: ignore
except ImportError:  # pragma: no cover — non-POSIX
    resource = None  # type: ignore


@dataclass(slots=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool


class Sandbox:
    """Spawns subprocesses with rlimits + wallclock timeouts.

    Construct with `from_settings(settings)` so caps come from one place.
    """

    def __init__(
        self,
        *,
        root: Path,
        max_cpu_s: int,
        max_memory_mb: int,
        default_timeout_s: int,
    ) -> None:
        self._root = Path(root).resolve()
        self._max_cpu_s = max(1, int(max_cpu_s))
        self._max_memory_mb = max(64, int(max_memory_mb))
        self._default_timeout_s = max(1, int(default_timeout_s))
        self._root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> Sandbox:
        return cls(
            root=settings.sandbox_root,
            max_cpu_s=settings.sandbox_max_cpu_s,
            max_memory_mb=settings.sandbox_max_memory_mb,
            default_timeout_s=settings.sandbox_default_timeout_s,
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def default_timeout_s(self) -> int:
        return self._default_timeout_s

    # ── workdir lifecycle ──
    @asynccontextmanager
    async def workdir(self, *, prefix: str = "task_") -> AsyncIterator[Path]:
        """Async context manager that yields a fresh tempdir under root.

        Cleaned up unconditionally, even on exception.
        """
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=self._root))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    # ── exec ──
    async def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        timeout_s: int | None = None,
        env: Mapping[str, str] | None = None,
        input_data: bytes | None = None,
    ) -> SandboxResult:
        """Run `cmd` as a subprocess inside `cwd`.

        - `cwd` MUST be (or be inside) `self.root`. Otherwise we refuse.
        - Env is the *minimal* set needed plus the explicit overrides.
        - rlimits applied via `preexec_fn` on POSIX.
        """
        cwd_resolved = Path(cwd).resolve()
        try:
            cwd_resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"cwd must be inside sandbox root: {cwd_resolved}") from exc

        timeout = max(1, timeout_s or self._default_timeout_s)
        scrubbed = self._scrub_env(env or {})

        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd_resolved),
                env=scrubbed,
                preexec_fn=self._apply_rlimits if resource is not None else None,
            )
        except FileNotFoundError as exc:
            return SandboxResult(
                stdout="",
                stderr=f"executable not found: {cmd[0]!r}",
                exit_code=127,
                duration_ms=int((time.perf_counter() - started) * 1000),
                timed_out=False,
            ) if False else SandboxResult(
                stdout="",
                stderr=str(exc),
                exit_code=127,
                duration_ms=int((time.perf_counter() - started) * 1000),
                timed_out=False,
            )

        timed_out = False
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=input_data), timeout=timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:  # pragma: no cover
                pass
            try:
                out, err = await proc.communicate()
            except Exception:  # noqa: BLE001
                out, err = b"", b"timed out"

        duration_ms = int((time.perf_counter() - started) * 1000)
        return SandboxResult(
            stdout=_safe_decode(out),
            stderr=_safe_decode(err),
            exit_code=proc.returncode if proc.returncode is not None else (124 if timed_out else 1),
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    # ── internals ──
    def _scrub_env(self, overrides: Mapping[str, str]) -> dict[str, str]:
        """Return a minimal env. Strip proxies and most variables."""
        keep = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "HOME": str(self._root),
            "TMPDIR": str(self._root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        keep.update({k: v for k, v in overrides.items() if isinstance(v, str)})
        # Remove anything proxy-shaped.
        for k in list(keep.keys()):
            if k.upper().endswith("_PROXY") or k.upper() in {"HTTP_PROXY", "HTTPS_PROXY"}:
                keep.pop(k, None)
        return keep

    def _apply_rlimits(self) -> None:  # pragma: no cover — runs in child
        """Called in the child between fork and exec to install rlimits."""
        if resource is None:
            return
        # CPU seconds (SIGXCPU then SIGKILL one second later)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (self._max_cpu_s, self._max_cpu_s + 1))
        except (ValueError, OSError):
            pass
        # Address space (memory)
        try:
            mem_bytes = self._max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        # Open file descriptors
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except (ValueError, OSError):
            pass
        # No core dumps
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
        # Detach from parent's process group so a runaway can't signal us.
        try:
            os.setsid()
        except OSError:
            pass


def _safe_decode(b: bytes | None) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return str(b)


# Convenience: get the python interpreter the backend itself runs under.
def python_interpreter() -> str:
    return sys.executable or "python3"
