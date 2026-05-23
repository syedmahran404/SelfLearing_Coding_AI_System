"""Tool framework + sandbox tools."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.base import BaseTool, Permission, ToolError
from app.tools.code_exec import PythonCodeTool
from app.tools.file_ops import FileListTool, FileReadTool, FileWriteTool
from app.tools.repo_analyzer import RepoAnalyzerTool
from app.tools.sandbox import Sandbox


@pytest.fixture
def sandbox(settings):
    return Sandbox.from_settings(settings)


def test_safe_path_rejects_escape(tmp_workdir: Path):
    with pytest.raises(ToolError):
        BaseTool.safe_path(tmp_workdir, "../escape")
    inside = BaseTool.safe_path(tmp_workdir, "ok.txt")
    assert str(inside).startswith(str(tmp_workdir))


def test_coerce_args_required_and_types():
    schema = {
        "type": "object",
        "required": ["x"],
        "properties": {"x": {"type": "integer"}, "y": {"type": "boolean"}},
    }
    with pytest.raises(ToolError):
        BaseTool.coerce_args({}, schema)
    out = BaseTool.coerce_args({"x": "5", "y": "true"}, schema)
    assert out["x"] == 5
    assert out["y"] is True


@pytest.mark.asyncio
async def test_file_write_then_read_roundtrip(tmp_workdir: Path):
    write = FileWriteTool()
    read = FileReadTool()
    from app.tools.base import ToolInput

    r1 = await write.run(
        ToolInput(
            args={"path": "a.txt", "content": "hello"},
            workdir=tmp_workdir,
            timeout_s=5,
        )
    )
    assert r1.ok and (tmp_workdir / "a.txt").read_text() == "hello"

    r2 = await read.run(
        ToolInput(args={"path": "a.txt"}, workdir=tmp_workdir, timeout_s=5)
    )
    assert r2.ok and r2.output["content"] == "hello"


@pytest.mark.asyncio
async def test_file_write_dryrun_does_not_touch_disk(tmp_workdir: Path):
    from app.tools.base import ToolInput

    write = FileWriteTool(dryrun=True)
    r = await write.run(
        ToolInput(
            args={"path": "b.txt", "content": "x"}, workdir=tmp_workdir, timeout_s=5
        )
    )
    assert r.ok
    assert r.output["dryrun"] is True
    assert not (tmp_workdir / "b.txt").exists()


@pytest.mark.asyncio
async def test_file_list_depth_capped(tmp_workdir: Path):
    (tmp_workdir / "a/b/c").mkdir(parents=True)
    (tmp_workdir / "a/b/c/d.txt").write_text("ok")
    (tmp_workdir / "top.txt").write_text("ok")

    from app.tools.base import ToolInput

    lister = FileListTool()
    r = await lister.run(
        ToolInput(
            args={"path": ".", "max_depth": 1}, workdir=tmp_workdir, timeout_s=5
        )
    )
    assert r.ok
    assert "top.txt" in r.output["entries"]
    # depth 1: a/ visible, but a/b/... not
    assert "a/" in r.output["entries"]
    assert not any(e.endswith("d.txt") for e in r.output["entries"])


@pytest.mark.asyncio
async def test_python_code_tool_runs_and_captures_stdout(sandbox, tmp_workdir):
    tool = PythonCodeTool(sandbox)
    from app.tools.base import ToolInput

    r = await tool.run(
        ToolInput(
            args={"source": "print('hi from sandbox')"},
            workdir=tmp_workdir,
            timeout_s=10,
        )
    )
    assert r.ok, r.error
    assert "hi from sandbox" in r.stdout


@pytest.mark.asyncio
async def test_python_code_tool_refuses_denylisted_pattern(sandbox, tmp_workdir):
    tool = PythonCodeTool(sandbox)
    from app.tools.base import ToolInput

    r = await tool.run(
        ToolInput(
            args={"source": "import shutil; shutil.rmtree('/'  )"},  # match pattern
            workdir=tmp_workdir,
            timeout_s=5,
        )
    )
    assert not r.ok
    assert "denylisted" in (r.error or "")


@pytest.mark.asyncio
async def test_python_code_tool_times_out(sandbox, tmp_workdir):
    tool = PythonCodeTool(sandbox)
    from app.tools.base import ToolInput

    r = await tool.run(
        ToolInput(
            args={"source": "import time; time.sleep(5)"},
            workdir=tmp_workdir,
            timeout_s=1,
        )
    )
    assert not r.ok
    assert (r.error or "").startswith("timed out") or r.exit_code != 0


@pytest.mark.asyncio
async def test_repo_analyzer_summarizes_files(tmp_workdir: Path):
    (tmp_workdir / "README.md").write_text("# hi")
    (tmp_workdir / "a.py").write_text("x = 1\n")

    tool = RepoAnalyzerTool()
    from app.tools.base import ToolInput

    r = await tool.run(
        ToolInput(args={"path": "."}, workdir=tmp_workdir, timeout_s=5)
    )
    assert r.ok
    assert r.output["totals"]["files"] >= 2
    assert "README.md" in r.output["anchors"]
