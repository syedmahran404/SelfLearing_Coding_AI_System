"""Orchestrator — the single state machine that walks a TaskGraph.

Lifecycle of a single run:
    classify → plan → for each subtask in topological order:
        build_context → run_specialist → run_tools? → evaluate
        if failed: reflect → maybe insert_subtasks or retry
    finalize_answer → memory_writes → finish_episode

Streaming
---------
`run_stream(ctx)` is an async generator yielding `OrchestratorEvent`s.
The HTTP layer transforms these into SSE chunks. Events are also written
to the tracer so an observer can reconstruct the run later.

Safety gates
------------
- `SAFETY_MAX_RECURSION_DEPTH` caps how often a subtask can be retried.
- `SAFETY_MIN_CONFIDENCE_TO_EXECUTE` blocks side-effecting tool runs whose
  upstream confidence is too low.
- `LLM_MAX_TOKENS_PER_TASK` bounds total tokens; we abort with PARTIAL.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from typing import TYPE_CHECKING

from app.agents import AgentRegistry
from app.agents.base import BaseAgent
from app.agents.planner import graph_from_metadata
from app.config import Settings
from app.memory.service import MemoryService

if TYPE_CHECKING:
    from app.learning.reflection_engine import ReflectionEngine
    from app.safety.guards import CircuitBreaker
from app.observability import Tracer, bind_trace, get_logger, new_trace_id, reset_trace
from app.observability.tracing import SpanKind, trace_span
from app.orchestration.router import IntentRouter
from app.orchestration.state import (
    RunState,
    RunStatus,
    SubtaskState,
    SubtaskStatus,
)
from app.rag.service import RagService
from app.schemas.agent_io import (
    AgentInput,
    AgentOutput,
    EvaluationResult,
    Reflection,
    Subtask,
    TaskIntent,
)
from app.tools.registry import ToolRegistry

logger = get_logger("orchestrator")


# ── streamed events ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class OrchestratorEvent:
    type: str  # "status" | "trace" | "delta" | "tool" | "memory" | "done" | "error"
    data: Any


@dataclass(slots=True)
class RunContext:
    """Per-request context handed to the orchestrator."""

    request: str
    db: AsyncSession
    user_id: UUID | None = None
    project_id: UUID | None = None
    session_id: UUID | None = None
    trace_id: str | None = None


class Orchestrator:
    """Walks a TaskGraph and drives agents; the only mutator of `RunState`."""

    def __init__(
        self,
        *,
        settings: Settings,
        agents: AgentRegistry,
        memory: MemoryService | None,
        rag: RagService | None,
        tools: ToolRegistry | None,
        reflection: "ReflectionEngine | None",
        tracer: Tracer,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self._settings = settings
        self._agents = agents
        self._memory = memory
        self._rag = rag
        self._tools = tools
        self._reflection = reflection
        self._tracer = tracer
        # Process-wide breaker keyed by (agent, intent). Trips after repeated
        # failures and refuses fresh runs for a cooldown — the last line of
        # defense against runaway reflection loops surviving across requests.
        from app.safety.guards import CircuitBreaker

        self._breaker: CircuitBreaker = circuit_breaker or CircuitBreaker(
            threshold=4, cooldown_s=60
        )

        # Lazy: we need a router; construct once.
        self._router: IntentRouter | None = None

    async def shutdown(self) -> None:
        return None

    # ── public entrypoint ───────────────────────────────────────────────
    async def run_stream(self, ctx: RunContext) -> AsyncIterator[OrchestratorEvent]:
        """Drive a full run; yield streaming events."""
        trace_id = ctx.trace_id or new_trace_id()
        token = bind_trace(trace_id)
        try:
            async for ev in self._run_stream_inner(ctx, trace_id):
                yield ev
        finally:
            reset_trace(token)

    async def _run_stream_inner(
        self, ctx: RunContext, trace_id: str
    ) -> AsyncIterator[OrchestratorEvent]:
        # 1) classify intent
        if self._router is None:
            llm = self._agents.get("planner")._llm  # type: ignore[attr-defined]
            assert llm is not None
            self._router = IntentRouter(llm)

        async with trace_span(self._tracer, "orchestrator.run", SpanKind.SYSTEM) as run_span:
            intent, intent_conf = await self._router.classify(ctx.request)
            state = RunState.new(
                ctx.request,
                intent=intent,
                trace_id=trace_id,
                user_id=ctx.user_id,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
            )
            run_span["payload"] = {
                "run_id": state.run_id,
                "intent": state.intent.value,
                "intent_confidence": intent_conf,
            }
            yield self._status(state, status=RunStatus.PLANNING)

            # Refuse the run if the breaker is open for this (agent, intent)
            # — protects across requests when a class of task keeps failing.
            if self._breaker.is_open("orchestrator", state.intent.value):
                msg = (
                    f"circuit breaker open for intent={state.intent.value}; "
                    "cooling down before accepting new work"
                )
                logger.warning(
                    "orchestrator_breaker_open",
                    run_id=state.run_id,
                    intent=state.intent.value,
                )
                state.notes.append(msg)
                state.status = RunStatus.ABORTED
                yield self._error(state, msg)
                yield self._done(state)
                return

            # 2) start an episode (if memory available)
            episode = None
            if self._memory is not None and ctx.user_id is not None:
                episode = await self._memory.start_episode(
                    ctx.db,
                    user_id=ctx.user_id,
                    intent=intent.value,
                    title=ctx.request[:120],
                    input_text=ctx.request,
                    project_id=ctx.project_id,
                    session_id=ctx.session_id,
                    trace_id=trace_id,
                )
                state.episode_id = episode.id

            # 3) plan
            try:
                plan_out = await self._invoke_agent(
                    "planner",
                    ai=self._make_input(state, ctx, request=ctx.request),
                    state=state,
                )
                state.plan = graph_from_metadata(plan_out.metadata)
            except Exception as exc:  # noqa: BLE001
                logger.exception("planner_failed", error=str(exc))
                yield self._error(state, f"planner failed: {exc}")
                state.status = RunStatus.FAILED
                yield self._done(state)
                return

            if state.plan is None or not state.plan.subtasks:
                yield self._error(state, "planner produced no subtasks")
                state.status = RunStatus.FAILED
                yield self._done(state)
                return

            state.subtasks = [SubtaskState(subtask=s) for s in state.plan.subtasks]
            yield OrchestratorEvent(type="plan", data=state.plan.model_dump(mode="json"))

            # 4) execute subtasks
            #
            # Recursion-safe driver:
            #   * `completed_ids` ensures every subtask is processed at most
            #     once, even if `_insert_subtasks` mutates `state.subtasks`
            #     under us (Defect 1: list mutation during iteration).
            #   * `state.max_iterations` is a global cap (Defect 2: no global
            #     iteration budget).
            #   * `state.insertions_remaining` budgets reflector-driven
            #     inserts (Defect 2 cont.).
            #   * `state.consecutive_subtask_failures` aborts a clearly stuck
            #     run before it drains tokens (Defect 3).
            state.status = RunStatus.RUNNING
            yield self._status(state)

            initial_n = len(state.subtasks)
            state.max_iterations = max(
                8,
                initial_n * 2 + self._settings.safety_max_recursion_depth * 2,
            )
            state.insertions_remaining = max(
                1, self._settings.safety_max_recursion_depth * 2
            )
            state.iterations = 0

            completed_ids: set[str] = set()
            while True:
                # Pick the first not-yet-completed subtask. Order-agnostic
                # relative to insertions, which is exactly the invariant we
                # need to make insertions safe.
                current = next(
                    (s for s in state.subtasks if s.subtask.id not in completed_ids),
                    None,
                )
                if current is None:
                    break  # all subtasks done

                if state.iterations >= state.max_iterations:
                    msg = (
                        f"aborted: orchestrator iteration cap reached "
                        f"({state.iterations}/{state.max_iterations})"
                    )
                    logger.warning(
                        "orchestrator_iteration_cap",
                        run_id=state.run_id,
                        iterations=state.iterations,
                        max_iterations=state.max_iterations,
                    )
                    state.notes.append(msg)
                    state.status = RunStatus.PARTIAL
                    break

                state.iterations += 1
                async for ev in self._run_subtask(state, ctx, current):
                    yield ev
                completed_ids.add(current.subtask.id)

                # Cross-subtask circuit breaker — guarantees we never burn
                # the full token budget on a clearly stuck run.
                if current.status == SubtaskStatus.PASSED:
                    state.consecutive_subtask_failures = 0
                elif current.status == SubtaskStatus.FAILED:
                    state.consecutive_subtask_failures += 1
                    if (
                        state.consecutive_subtask_failures
                        >= state.max_consecutive_failures
                    ):
                        msg = (
                            f"aborted: {state.consecutive_subtask_failures} "
                            "consecutive subtask failures"
                        )
                        logger.warning(
                            "orchestrator_consecutive_failures_tripped",
                            run_id=state.run_id,
                            failures=state.consecutive_subtask_failures,
                        )
                        state.notes.append(msg)
                        state.status = RunStatus.PARTIAL
                        break

                # Token budget abort.
                if state.tokens_used > self._settings.llm_max_tokens_per_task:
                    state.notes.append("aborted: token budget exceeded")
                    state.status = RunStatus.PARTIAL
                    break

            # 5) finalize answer
            state.final_answer = self._compose_final_answer(state)
            state.confidence = self._aggregate_confidence(state)
            if state.status not in (RunStatus.PARTIAL, RunStatus.FAILED):
                state.status = self._final_status(state)
            yield OrchestratorEvent(type="answer", data={"answer": state.final_answer})

            # 6) memory writes
            if self._memory is not None and ctx.user_id is not None:
                try:
                    async for ev in self._extract_memories(state, ctx):
                        yield ev
                except Exception as exc:  # noqa: BLE001
                    logger.warning("memory_writes_failed", error=str(exc))

            # 7) finish episode
            if episode is not None and self._memory is not None:
                try:
                    await self._memory.finish_episode(
                        ctx.db,
                        episode_id=state.episode_id,  # type: ignore[arg-type]
                        outcome=_outcome_for(state.status),
                        score=state.confidence,
                        confidence=state.confidence,
                        actions=state.actions,
                        summary=_episode_summary(state),
                        tokens_used=state.tokens_used,
                        cost_usd=state.cost_usd,
                        duration_ms=state.duration_ms,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("finish_episode_failed", error=str(exc))

            state.finished_at = time.time()
            # Update the cross-request breaker on the way out.
            if state.status in (RunStatus.SUCCESS, RunStatus.PARTIAL):
                self._breaker.record_success("orchestrator", state.intent.value)
            else:
                self._breaker.record_failure("orchestrator", state.intent.value)
            yield self._done(state)

    # ── per-subtask loop ────────────────────────────────────────────────
    async def _run_subtask(
        self,
        state: RunState,
        ctx: RunContext,
        s: SubtaskState,
    ) -> AsyncIterator[OrchestratorEvent]:
        max_attempts = max(1, min(s.subtask.max_attempts, self._settings.safety_max_recursion_depth))
        for attempt in range(1, max_attempts + 1):
            s.attempts = attempt
            s.status = SubtaskStatus.RUNNING
            yield OrchestratorEvent(
                type="subtask",
                data={
                    "id": s.subtask.id,
                    "title": s.subtask.title,
                    "agent": s.subtask.agent,
                    "attempt": attempt,
                    "status": s.status.value,
                },
            )

            # 1) build context
            ctx_out = await self._invoke_agent(
                "context_builder",
                ai=self._make_input(state, ctx, subtask=s.subtask),
                state=state,
            )
            built_context = ctx_out.metadata.get("context") or {}

            # 2) specialist
            try:
                specialist_out = await self._invoke_agent(
                    s.subtask.agent,
                    ai=self._make_input(state, ctx, subtask=s.subtask, context=built_context),
                    state=state,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("specialist_failed", agent=s.subtask.agent, error=str(exc))
                state.failures.append(f"{s.subtask.agent}: {exc}")
                s.status = SubtaskStatus.FAILED
                break
            s.last_output = specialist_out

            # 3) tool execution if requested
            tool_runs: list[dict[str, Any]] = []
            if specialist_out.tool_calls:
                if specialist_out.confidence < self._settings.safety_min_confidence_to_execute:
                    state.notes.append(
                        f"tool_calls skipped (low confidence {specialist_out.confidence:.2f})"
                    )
                else:
                    if self._tools is not None and self._agents.list().count("tool_executor"):
                        ext_out = await self._invoke_agent(
                            "tool_executor",
                            ai=self._make_input(
                                state,
                                ctx,
                                subtask=s.subtask,
                                context=built_context,
                                extras_extra={
                                    "tool_calls": [
                                        t.model_dump(mode="json") for t in specialist_out.tool_calls
                                    ]
                                },
                            ),
                            state=state,
                        )
                        tool_runs = ext_out.metadata.get("tool_runs") or []
                        s.last_tool_runs = tool_runs
                        for tr in tool_runs:
                            yield OrchestratorEvent(type="tool", data=tr)

            # 4) evaluate
            eval_out = await self._invoke_agent(
                "evaluator",
                ai=self._make_input(
                    state,
                    ctx,
                    subtask=s.subtask,
                    context=built_context,
                    extras_extra={
                        "agent_output_summary": specialist_out.summary,
                        "agent_answer": specialist_out.answer or "",
                        "tool_runs": tool_runs,
                    },
                ),
                state=state,
            )
            evaluation = eval_out.metadata.get("evaluation") or {}
            s.last_evaluation = evaluation
            try:
                ev = EvaluationResult.model_validate(evaluation) if evaluation else None
            except Exception:  # noqa: BLE001
                ev = None
            yield OrchestratorEvent(type="evaluation", data=evaluation)

            passed = bool(ev and ev.passed)
            score = float(ev.score) if ev else 0.0

            state.record_action(
                "subtask_attempt",
                {
                    "subtask_id": s.subtask.id,
                    "attempt": attempt,
                    "agent": s.subtask.agent,
                    "passed": passed,
                    "score": score,
                    "summary": specialist_out.summary,
                },
            )

            if passed:
                s.status = SubtaskStatus.PASSED
                yield OrchestratorEvent(
                    type="subtask",
                    data={
                        "id": s.subtask.id,
                        "status": s.status.value,
                        "attempts": attempt,
                        "score": score,
                    },
                )
                return

            # 5) failed → reflect (if not last attempt)
            if attempt < max_attempts:
                state.status = RunStatus.REFLECTING
                yield self._status(state)
                refl_out = await self._invoke_agent(
                    "reflector",
                    ai=self._make_input(
                        state,
                        ctx,
                        subtask=s.subtask,
                        context=built_context,
                        extras_extra={
                            "agent_output_summary": specialist_out.summary,
                            "tool_runs": tool_runs,
                            "failures": (ev.failures if ev else ["evaluation failed"]),
                        },
                    ),
                    state=state,
                )
                reflection_d = refl_out.metadata.get("reflection") or {}
                s.last_reflection = reflection_d
                try:
                    refl = Reflection.model_validate(reflection_d) if reflection_d else None
                except Exception:  # noqa: BLE001
                    refl = None
                yield OrchestratorEvent(type="reflection", data=reflection_d)

                # Insert reflector-suggested new subtasks before this one's retry.
                if refl and refl.new_subtasks:
                    self._insert_subtasks(state, before=s, new_subtasks=refl.new_subtasks)
                state.status = RunStatus.RUNNING
                yield self._status(state)
                continue  # retry

            # final failure
            s.status = SubtaskStatus.FAILED
            state.failures.append(
                f"subtask {s.subtask.id} ({s.subtask.title}) failed after {attempt} attempts"
            )
            yield OrchestratorEvent(
                type="subtask",
                data={
                    "id": s.subtask.id,
                    "status": s.status.value,
                    "attempts": attempt,
                    "score": score,
                },
            )

    # ── memory extraction ───────────────────────────────────────────────
    async def _extract_memories(
        self, state: RunState, ctx: RunContext
    ) -> AsyncIterator[OrchestratorEvent]:
        if self._memory is None or ctx.user_id is None:
            return
        last_reflection_lesson = ""
        for s in reversed(state.subtasks):
            if s.last_reflection and s.last_reflection.get("lesson"):
                last_reflection_lesson = s.last_reflection["lesson"]
                break

        try:
            ma_out = await self._invoke_agent(
                "memory_agent",
                ai=self._make_input(
                    state,
                    ctx,
                    extras_extra={
                        "outcome": state.status.value,
                        "final_answer": state.final_answer,
                        "reflection_lesson": last_reflection_lesson,
                    },
                ),
                state=state,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_agent_failed", error=str(exc))
            return

        for w in ma_out.memory_writes:
            try:
                mem, created = await self._memory.remember(
                    ctx.db,
                    user_id=ctx.user_id,
                    kind=w.kind,
                    content=w.content,
                    tags=w.tags,
                    project_id=(ctx.project_id if w.project_scoped else None),
                    confidence=w.confidence,
                )
                yield OrchestratorEvent(
                    type="memory",
                    data={
                        "id": str(mem.id),
                        "kind": mem.kind,
                        "content": mem.content[:300],
                        "created": created,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory_write_failed", error=str(exc))

    # ── helpers ─────────────────────────────────────────────────────────
    def _make_input(
        self,
        state: RunState,
        ctx: RunContext,
        *,
        request: str | None = None,
        subtask: Subtask | None = None,
        context: dict | None = None,
        extras_extra: dict[str, Any] | None = None,
    ) -> AgentInput:
        from app.schemas.agent_io import BudgetedContext

        ai_context: BudgetedContext | None = None
        if context is not None:
            try:
                ai_context = BudgetedContext.model_validate(context)
            except Exception:  # noqa: BLE001
                ai_context = None

        extras: dict[str, Any] = {"db_session": ctx.db}
        if extras_extra:
            extras.update(extras_extra)

        return AgentInput(
            run_id=state.run_id,
            trace_id=state.trace_id,
            user_id=state.user_id,
            project_id=state.project_id,
            session_id=state.session_id,
            intent=state.intent,
            request=request or state.request,
            subtask=subtask,
            context=ai_context or BudgetedContext(),
            extras=extras,
        )

    async def _invoke_agent(
        self, name: str, *, ai: AgentInput, state: RunState
    ) -> AgentOutput:
        agent: BaseAgent
        try:
            agent = self._agents.get(name)
        except KeyError:
            # An agent the planner referenced isn't present. Fall back to coder.
            logger.warning("agent_missing_fallback", requested=name)
            agent = self._agents.get("coder")

        out = await agent.run(ai)
        # Track token + cost from the trace span the agent emitted.
        # (BaseAgent fills span tokens; we rely on the meter aggregation.)
        return out

    def _compose_final_answer(self, state: RunState) -> str:
        # Prefer the last researcher/coder output's `answer` if any.
        for s in reversed(state.subtasks):
            if s.last_output and s.last_output.answer:
                return s.last_output.answer
        # Otherwise compose from summaries.
        bullets: list[str] = []
        for s in state.subtasks:
            if s.last_output and s.last_output.summary:
                tag = "✓" if s.status == SubtaskStatus.PASSED else (
                    "✗" if s.status == SubtaskStatus.FAILED else "·"
                )
                bullets.append(f"{tag} {s.subtask.title}: {s.last_output.summary}")
        if bullets:
            return "\n".join(bullets)
        return "(no answer produced)"

    def _aggregate_confidence(self, state: RunState) -> float:
        if not state.subtasks:
            return 0.0
        weights: list[float] = []
        for s in state.subtasks:
            c = (s.last_output.confidence if s.last_output else 0.0)
            if s.status == SubtaskStatus.PASSED:
                weights.append(min(1.0, c + 0.1))
            elif s.status == SubtaskStatus.FAILED:
                weights.append(max(0.0, c - 0.2))
            else:
                weights.append(c * 0.6)
        return sum(weights) / len(weights)

    def _final_status(self, state: RunState) -> RunStatus:
        passed = sum(1 for s in state.subtasks if s.status == SubtaskStatus.PASSED)
        failed = sum(1 for s in state.subtasks if s.status == SubtaskStatus.FAILED)
        total = len(state.subtasks)
        if failed == 0 and passed == total:
            return RunStatus.SUCCESS
        if passed > 0 and passed >= failed:
            return RunStatus.PARTIAL
        return RunStatus.FAILED

    def _insert_subtasks(
        self, state: RunState, *, before: SubtaskState, new_subtasks: list[Subtask]
    ) -> None:
        # Honor the per-run insertion budget. Without this cap, a reflector
        # that keeps proposing new_subtasks for failing subtasks could grow
        # state.subtasks unboundedly even though each individual subtask has
        # a max_attempts cap. (Defect 2 in the recursion analysis.)
        if not new_subtasks:
            return
        if state.insertions_remaining <= 0:
            logger.warning(
                "subtask_insert_budget_exhausted",
                run_id=state.run_id,
                dropped=len(new_subtasks),
            )
            state.notes.append(
                f"reflector new_subtasks suppressed (budget exhausted): "
                f"{len(new_subtasks)} dropped"
            )
            return

        accepted = list(new_subtasks)[: state.insertions_remaining]
        state.insertions_remaining -= len(accepted)
        idx = state.subtasks.index(before)
        for offset, ns in enumerate(accepted):
            state.subtasks.insert(idx + offset, SubtaskState(subtask=ns))

    # ── event constructors ──
    def _status(self, state: RunState, *, status: RunStatus | None = None) -> OrchestratorEvent:
        if status is not None:
            state.status = status
        return OrchestratorEvent(
            type="status",
            data={"run_id": state.run_id, "status": state.status.value},
        )

    def _error(self, state: RunState, message: str) -> OrchestratorEvent:
        return OrchestratorEvent(type="error", data={"run_id": state.run_id, "message": message})

    def _done(self, state: RunState) -> OrchestratorEvent:
        return OrchestratorEvent(type="done", data=state.to_dict())


# ── helpers ───────────────────────────────────────────────────────────────


def _outcome_for(status: RunStatus) -> str:
    if status == RunStatus.SUCCESS:
        return "success"
    if status == RunStatus.PARTIAL:
        return "partial"
    return "failure"


def _episode_summary(state: RunState) -> str:
    head = f"[{state.intent.value}/{state.status.value}] {state.request[:160]}"
    actions = state.actions[-6:]
    body = "\n".join(
        f"- {a['kind']}: {a['payload'].get('summary') or a['payload'].get('subtask_id', '?')}"
        for a in actions
    )
    return f"{head}\n{body}"
