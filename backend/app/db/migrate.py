"""Bootstrap migration runner.

Usage:
    python -m app.db.migrate                # create all tables
    python -m app.db.migrate --drop         # drop all (dev only)

For real evolutions, switch to Alembic — `alembic` is already declared in
pyproject. This module exists so a fresh dev or CI environment is one
command away from a usable schema.
"""
from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings
from app.db.session import Base, init_engine, shutdown_engine
from app.observability.logger import configure_logging, get_logger

# Importing models registers them on Base.metadata (do not remove).
from app.db import models as _models  # noqa: F401


logger = get_logger("db.migrate")


async def create_all() -> None:
    settings = get_settings()
    engine = init_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("schema_create_all_complete", tables=len(Base.metadata.tables))


async def drop_all() -> None:
    settings = get_settings()
    engine = init_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("schema_drop_all_complete")


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DB schema migration")
    p.add_argument("--drop", action="store_true", help="drop all tables (dev only)")
    args = p.parse_args(argv)
    configure_logging("INFO")
    try:
        if args.drop:
            await drop_all()
        await create_all()
        return 0
    finally:
        await shutdown_engine()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
