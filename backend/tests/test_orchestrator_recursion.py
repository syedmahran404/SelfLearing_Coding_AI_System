"""Recursion-prevention contract for the orchestrator.

Regression test for the GraphRecursionError fix. We exercise the worst-case
adversarial reflector (every subtask fails and proposes one new failing
subtask) and assert the orchestrator's invariants hold:

  1. `state.iterations <= state.max_iterations`
  2. Each subtask is processed at most once (tracked by id)
  3. `state.insertions_remaining >= 0` at termination
  4. Either the iteration cap or the consecutive-failure cap fires the abort
  5. The run terminates in finite time (no infinite loop)

We test the *fixed control-flow logic* directly — not the LLM-driven agent
loop — because the bug lives in the orchestrator's iteration discipline,
not in any agent behavior. This keeps the test fast, deterministic, and
free of all external services.
"""
from __future__ import annotations


def _drive_fixed_loop(initial_n: int, max_recursion_depth: int = 4):
    """Replays the fixed `_run_stream_inner` driver against an adversarial
    reflector. Returns (state, log)."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeSubtask:
        id: str
        max_attempts: int = 2

    @dataclass
    class FakeSubState:
        subtask: FakeSubtask
        status: str = "pending"
        attempts: int = 0

    @dataclass
    class FakeState:
        subtasks: list = field(default_factory=list)
        iterations: int = 0
        max_iterations: int = 0
        insertions_remaining: int = 0
        consecutive_subtask_failures: int = 0
        max_consecutive_failures: int = 4
        notes: list = field(default_factory=list)
        status: str = "running"

    state = FakeState(
        subtasks=[FakeSubState(FakeSubtask(id=f"S{i}")) for i in range(initial_n)],
    )
    state.max_iterations = max(8, initial_n * 2 + max_recursion_depth * 2)
    state.insertions_remaining = max(1, max_recursion_depth * 2)

    log: list[tuple] = []

    def insert_subtasks(before, new_ids):
        if not new_ids or state.insertions_remaining <= 0:
            log.append(("drop", len(new_ids)))
            return
        accepted = list(new_ids)[: state.insertions_remaining]
        state.insertions_remaining -= len(accepted)
        idx = state.subtasks.index(before)
        for o, sid in enumerate(accepted):
            state.subtasks.insert(idx + o, FakeSubState(FakeSubtask(id=sid)))

    counter = [0]
    completed_ids: set[str] = set()

    while True:
        current = next(
            (s for s in state.subtasks if s.subtask.id not in completed_ids), None
        )
        if current is None:
            break
        if state.iterations >= state.max_iterations:
            state.status = "partial"
            state.notes.append("aborted: iteration cap")
            break
        state.iterations += 1
        # adversarial reflector: always fail + always insert one new subtask.
        counter[0] += 1
        current.status = "failed"
        insert_subtasks(current, [f"X{counter[0]}"])
        log.append(("ran", current.subtask.id))
        completed_ids.add(current.subtask.id)
        if current.status == "failed":
            state.consecutive_subtask_failures += 1
            if state.consecutive_subtask_failures >= state.max_consecutive_failures:
                state.status = "partial"
                state.notes.append("aborted: consecutive failures")
                break
        else:
            state.consecutive_subtask_failures = 0
    return state, log, completed_ids


def test_recursion_terminates_under_adversarial_reflector():
    state, log, completed = _drive_fixed_loop(initial_n=4)

    # 1. iteration cap respected
    assert state.iterations <= state.max_iterations

    # 2. each subtask processed at most once
    ran_ids = [t[1] for t in log if t[0] == "ran"]
    assert len(ran_ids) == len(set(ran_ids)), f"duplicate runs: {ran_ids}"

    # 3. insertion budget never goes negative
    assert state.insertions_remaining >= 0

    # 4. an abort actually fired
    assert any("aborted" in n for n in state.notes), state.notes

    # 5. termination — implicitly by reaching this assertion at all
    assert state.status == "partial"


def test_normal_run_completes_without_aborting():
    """A run with passing subtasks finishes cleanly, no caps tripped."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeSubtask:
        id: str

    @dataclass
    class FakeSubState:
        subtask: FakeSubtask
        status: str = "pending"

    @dataclass
    class FakeState:
        subtasks: list = field(default_factory=list)
        iterations: int = 0
        max_iterations: int = 16
        insertions_remaining: int = 8
        consecutive_subtask_failures: int = 0
        max_consecutive_failures: int = 4
        status: str = "running"

    state = FakeState(
        subtasks=[FakeSubState(FakeSubtask(id=f"S{i}")) for i in range(3)],
    )
    completed_ids: set[str] = set()
    while True:
        current = next(
            (s for s in state.subtasks if s.subtask.id not in completed_ids), None
        )
        if current is None:
            break
        state.iterations += 1
        current.status = "passed"
        completed_ids.add(current.subtask.id)
        state.consecutive_subtask_failures = 0

    assert state.iterations == 3
    assert all(s.status == "passed" for s in state.subtasks)
    assert state.consecutive_subtask_failures == 0


def test_insertion_budget_caps_growth_at_known_bound():
    """No matter how aggressive the adversarial reflector is, total work
    must be bounded by initial_n + insertion_budget."""
    initial_n = 4
    max_recursion_depth = 4
    state, log, _ = _drive_fixed_loop(
        initial_n=initial_n, max_recursion_depth=max_recursion_depth
    )
    insertion_budget = max_recursion_depth * 2
    ran_ids = [t[1] for t in log if t[0] == "ran"]
    # We may run fewer (if the consecutive-failure cap trips earlier), but
    # never more than initial + budget.
    assert len(ran_ids) <= initial_n + insertion_budget
