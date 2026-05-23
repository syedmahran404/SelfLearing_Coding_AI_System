"""Composition root.

Builds a `Runtime` — the typed container of every long-lived service the
application uses (LLM provider, memory subsystem, RAG service, tool
registry, agent registry, orchestrator). The lifespan handler in `main.py`
constructs exactly one Runtime and stashes it on `app.state.runtime`.

Subsequent tasks fill in:
    * llm provider (task #4)
    * memory + RAG (tasks #6, #7)
    * tools (task #8)
    * agents + orchestrator (tasks #9, #10)
    * learning engine (task #11)
    * project indexer (task #12)

All slots are typed `Optional` here so imports stay valid before each
subsystem lands. As subsystems land they are wired in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from app.config import Settings
from app.db.qdrant import QdrantStore, get_qdrant
from app.db.redis_client import RedisClient, get_redis
from app.observability import Tracer, get_logger

if TYPE_CHECKING:  # avoid hard imports during early bootstrap
    from app.agents.base import AgentRegistry
    from app.learning.confidence import ConfidenceCalibrator
    from app.learning.pattern_evolver import PatternEvolver
    from app.learning.reflection_engine import ReflectionEngine
    from app.llm.provider import LLMProvider
    from app.memory.service import MemoryService
    from app.orchestration.orchestrator import Orchestrator
    from app.project.indexer import ProjectIndexer
    from app.rag.service import RagService
    from app.tools.registry import ToolRegistry

logger = get_logger("bootstrap")


@dataclass
class Runtime:
    """Typed service container shared across the request lifecycle."""

    settings: Settings
    tracer: Tracer
    redis: RedisClient
    qdrant: QdrantStore

    # Filled in as subsystems land. Optional so partial boots are valid.
    llm: Optional["LLMProvider"] = None
    memory: Optional["MemoryService"] = None
    rag: Optional["RagService"] = None
    tools: Optional["ToolRegistry"] = None
    agents: Optional["AgentRegistry"] = None
    orchestrator: Optional["Orchestrator"] = None
    reflection: Optional["ReflectionEngine"] = None
    pattern_evolver: Optional["PatternEvolver"] = None
    confidence: Optional["ConfidenceCalibrator"] = None
    project_indexer: Optional["ProjectIndexer"] = None

    # Free-form extension slot for tests / plugins.
    extras: dict[str, Any] = field(default_factory=dict)

    async def shutdown(self) -> None:
        """Best-effort shutdown of components that own background tasks."""
        for name in ("orchestrator", "memory", "rag"):
            obj = getattr(self, name, None)
            if obj is not None and hasattr(obj, "shutdown"):
                try:
                    await obj.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("runtime_shutdown_component_failed", component=name, error=str(exc))


async def build_runtime(settings: Settings, tracer: Tracer) -> Runtime:
    """Construct the Runtime by wiring the subsystems that have landed.

    Order matters: lower layers must be ready before higher ones reference
    them. The composition order mirrors the dependency graph:
        config → infra clients → llm → memory → rag → tools → agents → orchestrator
    """
    redis = get_redis()
    qdrant = get_qdrant()

    runtime = Runtime(settings=settings, tracer=tracer, redis=redis, qdrant=qdrant)

    # Wire LLM provider.
    try:
        from app.llm import build_llm_provider

        runtime.llm = build_llm_provider(settings)
        logger.info("runtime_llm_ready", provider=settings.llm_provider)
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_llm_skipped", error=str(exc))

    # Wire memory.
    if runtime.llm is not None:
        try:
            from app.memory.service import MemoryService

            runtime.memory = MemoryService(
                settings=settings,
                redis=redis,
                qdrant=qdrant,
                llm=runtime.llm,
                tracer=tracer,
            )
            logger.info("runtime_memory_ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_memory_skipped", error=str(exc))

    # Wire RAG.
    if runtime.llm is not None:
        try:
            from app.rag.service import RagService

            runtime.rag = RagService(
                settings=settings,
                qdrant=qdrant,
                llm=runtime.llm,
                tracer=tracer,
            )
            logger.info("runtime_rag_ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_rag_skipped", error=str(exc))

    # Wire tools.
    try:
        from app.tools.registry import build_default_registry

        runtime.tools = build_default_registry(settings)
        logger.info("runtime_tools_ready", tool_count=len(runtime.tools.list()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_tools_skipped", error=str(exc))

    # Wire agents.
    if runtime.llm is not None and runtime.memory is not None:
        try:
            from app.agents import build_agent_registry

            runtime.agents = build_agent_registry(
                settings=settings,
                llm=runtime.llm,
                memory=runtime.memory,
                rag=runtime.rag,
                tools=runtime.tools,
                tracer=tracer,
            )
            logger.info("runtime_agents_ready", agent_count=len(runtime.agents.list()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_agents_skipped", error=str(exc))

    # Wire reflection engine.
    if runtime.memory is not None and runtime.llm is not None:
        try:
            from app.learning.reflection_engine import ReflectionEngine

            runtime.reflection = ReflectionEngine(
                memory=runtime.memory, llm=runtime.llm, tracer=tracer
            )
            logger.info("runtime_reflection_ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_reflection_skipped", error=str(exc))

    # Wire pattern evolver.
    if runtime.llm is not None:
        try:
            from app.learning.pattern_evolver import PatternEvolver

            runtime.pattern_evolver = PatternEvolver(llm=runtime.llm, tracer=tracer)
            logger.info("runtime_pattern_evolver_ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_pattern_evolver_skipped", error=str(exc))

    # Wire confidence calibrator.
    try:
        from app.learning.confidence import ConfidenceCalibrator

        runtime.confidence = ConfidenceCalibrator()
        logger.info("runtime_confidence_ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_confidence_skipped", error=str(exc))

    # Wire orchestrator (top of the stack).
    if runtime.agents is not None:
        try:
            from app.orchestration.orchestrator import Orchestrator

            runtime.orchestrator = Orchestrator(
                settings=settings,
                agents=runtime.agents,
                memory=runtime.memory,
                rag=runtime.rag,
                tools=runtime.tools,
                reflection=runtime.reflection,
                tracer=tracer,
            )
            logger.info("runtime_orchestrator_ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_orchestrator_skipped", error=str(exc))

    # Wire project indexer.
    try:
        from app.project.indexer import ProjectIndexer

        runtime.project_indexer = ProjectIndexer(
            settings=settings, qdrant=qdrant, llm=runtime.llm, tracer=tracer
        )
        logger.info("runtime_project_indexer_ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_project_indexer_skipped", error=str(exc))

    return runtime
