"""Staged-write rollback area.

Pattern:
    rba = RollbackArea(workdir)
    rba.stage("foo.py", new_content)
    rba.stage("bar/baz.py", other_content)
    if eval_passed:
        rba.promote()        # writes everything atomically (best-effort)
    else:
        rba.discard()        # zero side effects

Used by the orchestrator when running a coder/debugger subtask: changes
are staged, then promoted only on a passing evaluation.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.observability import get_logger

logger = get_logger("safety.rollback")


@dataclass(slots=True)
class StagedFile:
    target_rel_path: str
    staged_abs_path: str
    bytes: int


@dataclass(slots=True)
class RollbackArea:
    """Staging area for files we plan to write atomically."""

    workdir: Path
    staging_dir: Path = field(init=False)
    staged: list[StagedFile] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.staging_dir = Path(tempfile.mkdtemp(prefix="staged_", dir=self.workdir))

    def stage(self, target_rel_path: str, content: str) -> StagedFile:
        rel = target_rel_path.lstrip("/")
        staged_path = self.staging_dir / rel
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_text(content, encoding="utf-8")
        sf = StagedFile(
            target_rel_path=rel,
            staged_abs_path=str(staged_path),
            bytes=len(content.encode("utf-8")),
        )
        self.staged.append(sf)
        return sf

    def promote(self) -> list[str]:
        """Move staged files into the workdir. Best-effort atomicity per file
        (rename within the same filesystem). Returns the list of promoted
        rel paths."""
        promoted: list[str] = []
        for sf in self.staged:
            target = self.workdir / sf.target_rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                # os.replace is atomic on POSIX when source/target are on
                # the same filesystem (true here — both under workdir).
                os.replace(sf.staged_abs_path, target)
                promoted.append(sf.target_rel_path)
            except OSError as exc:
                logger.warning(
                    "rollback_promote_failed",
                    src=sf.staged_abs_path,
                    dst=str(target),
                    error=str(exc),
                )
        # Cleanup staging dir even on partial success.
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self.staged.clear()
        return promoted

    def discard(self) -> None:
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self.staged.clear()
        logger.info("rollback_discarded", workdir=str(self.workdir))
