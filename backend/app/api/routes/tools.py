"""/tools — list registered tools and (admin) invoke one ad-hoc."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import require_tools
from app.bootstrap import Runtime

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def list_tools(rt: Runtime = Depends(require_tools)) -> dict[str, Any]:
    schemas = rt.tools.list()  # type: ignore[union-attr]
    return {
        "tools": [
            {
                "name": s.name,
                "description": s.description,
                "permissions": [p.value for p in s.permissions],
                "input_schema": s.input_schema,
                "output_schema": s.output_schema,
                "default_timeout_s": s.default_timeout_s,
                "safe_default": s.safe_default,
            }
            for s in schemas
        ]
    }


class ToolInvokeRequest(BaseModel):
    args: dict[str, Any] = {}
    timeout_s: int | None = None


@router.post("/{tool_name}/invoke")
async def invoke_tool(
    tool_name: str,
    body: ToolInvokeRequest,
    rt: Runtime = Depends(require_tools),
) -> dict[str, Any]:
    """Ad-hoc invocation. Useful for debugging tool schemas + sandboxing.

    A fresh workdir is created per invocation; outputs are returned but no
    files are persisted.
    """
    try:
        result = await rt.tools.invoke(  # type: ignore[union-attr]
            tool_name, body.args, timeout_s=body.timeout_s
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc))
    return result.to_dict()
