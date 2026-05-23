"""Agent layer.

Each agent has:
- a narrow responsibility (`responsibility` attribute)
- a focused system prompt (`prompts.py`)
- a typed I/O contract (AgentInput → AgentOutput)
- an internal call graph that is *bounded* (no agent-to-agent calls; only
  the orchestrator routes between agents)

`AgentRegistry` is a dict-of-agents shipped to the orchestrator and the
ToolExecutor agent. Agents are constructed in `build_agent_registry` —
the single composition point.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import AgentRegistry, BaseAgent
from app.agents.coder import CoderAgent
from app.agents.context_builder import ContextBuilderAgent
from app.agents.debugger import DebuggerAgent
from app.agents.decomposer import TaskDecomposerAgent
from app.agents.evaluator import EvaluatorAgent
from app.agents.memory_agent import MemoryAgent
from app.agents.planner import PlannerAgent
from app.agents.reflector import ReflectorAgent
from app.agents.researcher import ResearcherAgent
from app.agents.tool_executor import ToolExecutorAgent

if TYPE_CHECKING:
    from app.config import Settings
    from app.llm.provider import LLMProvider
    from app.memory.service import MemoryService
    from app.observability import Tracer
    from app.rag.service import RagService
    from app.tools.registry import ToolRegistry


__all__ = [
    "AgentRegistry",
    "BaseAgent",
    "PlannerAgent",
    "TaskDecomposerAgent",
    "ContextBuilderAgent",
    "ResearcherAgent",
    "CoderAgent",
    "DebuggerAgent",
    "ToolExecutorAgent",
    "EvaluatorAgent",
    "ReflectorAgent",
    "MemoryAgent",
    "build_agent_registry",
]


def build_agent_registry(
    *,
    settings: "Settings",
    llm: "LLMProvider",
    memory: "MemoryService",
    rag: "RagService | None",
    tools: "ToolRegistry | None",
    tracer: "Tracer",
) -> AgentRegistry:
    """Construct every default agent and put it in a registry.

    Order is by dependency: agents that don't need the registry first; agents
    that read from it (none of ours do, but that's the rule) last.
    """
    reg = AgentRegistry()

    reg.register(PlannerAgent(llm=llm, memory=memory, tracer=tracer, settings=settings))
    reg.register(TaskDecomposerAgent(llm=llm, tracer=tracer, settings=settings))
    reg.register(
        ContextBuilderAgent(memory=memory, rag=rag, tracer=tracer, settings=settings)
    )
    reg.register(ResearcherAgent(llm=llm, rag=rag, tracer=tracer, settings=settings))
    reg.register(CoderAgent(llm=llm, tracer=tracer, settings=settings))
    reg.register(DebuggerAgent(llm=llm, tracer=tracer, settings=settings))
    if tools is not None:
        reg.register(ToolExecutorAgent(tools=tools, tracer=tracer, settings=settings))
    reg.register(EvaluatorAgent(llm=llm, tools=tools, tracer=tracer, settings=settings))
    reg.register(ReflectorAgent(llm=llm, tracer=tracer, settings=settings))
    reg.register(MemoryAgent(llm=llm, memory=memory, tracer=tracer, settings=settings))

    return reg
