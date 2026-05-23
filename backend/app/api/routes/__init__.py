"""HTTP routes — one module per resource group.

`register_routes(app)` is the single seam between the FastAPI app and the
route modules. New route modules should be added here once and forgotten.
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import health


def register_routes(app: FastAPI) -> None:
    """Mount all route modules onto the given app."""
    app.include_router(health.router)

    # Conditionally include routers that depend on subsystems being ready.
    # Each module's own import handles missing deps gracefully.
    for modname in ("chat", "projects", "memory", "tools", "agents", "traces", "metrics"):
        try:
            module = __import__(f"app.api.routes.{modname}", fromlist=["router"])
        except Exception:  # noqa: BLE001
            # The route module hasn't landed yet; skip cleanly.
            continue
        router = getattr(module, "router", None)
        if router is not None:
            app.include_router(router)
