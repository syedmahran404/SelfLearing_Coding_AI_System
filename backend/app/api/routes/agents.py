"""/agents — describe the registered specialists."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_runtime
from app.bootstrap import Runtime

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("")
async def list_agents(rt: Runtime = Depends(get_runtime)) -> dict:
    if rt.agents is None:
        return {"agents": []}
    return {"agents": rt.agents.describe()}
