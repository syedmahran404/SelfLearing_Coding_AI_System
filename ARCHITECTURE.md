# Self-Learning Coding AI — Architecture

> A production-oriented autonomous coding system that plans, codes, debugs, reflects,
> and **learns from its own outcomes** while keeping safety, cost, and context under
> tight control.

This document is the source of truth for *why* the system is built the way it is.
Every component in `backend/app/` traces back to a decision recorded here.

---

## 1. Design goals & non-goals

### Goals
1. **Adaptive intelligence**: the system should perform better next week than this week
   on the same class of tasks, without code changes from a human, by accumulating
   lessons and patterns into a queryable knowledge base.
2. **Bounded autonomy**: every autonomous loop has an explicit termination condition,
   a confidence floor, and a rollback path. No "agent does whatever it wants forever".
3. **Token & memory efficiency**: context is *constructed*, not *concatenated*.
   The hot path never trusts "just dump everything into the prompt".
4. **Modular extensibility**: agents, tools, memory backends, and LLM providers are
   all behind narrow interfaces so any one of them can be swapped without ripples.
5. **Observability first**: every agent call, tool run, retrieval, and reflection
   produces a structured trace event. Debugging an autonomous system without traces
   is hopeless.

### Non-goals
- Replacing fine-tuning. We learn through *retrieved memory*, not weight updates.
- General-purpose agent framework. We optimize specifically for *coding* workflows.
- Cloud-vendor coupling. The system runs on docker-compose; cloud is an exercise.

---

## 2. Systems analysis: what kills systems like this

Before designing components I enumerated the failure modes that historically destroy
self-improving agent systems. The architecture is shaped to make each of these
hard or impossible by construction.

| # | Failure mode | Mitigation |
|---|---|---|
| F1 | **Context pollution** — irrelevant memories crowd out signal | Retrieval is scored, ranked, and budgeted; `ContextBuilder` enforces a token budget per section |
| F2 | **Hallucinated APIs** — agent invents library functions | `safety.validators.HallucinationGuard` cross-checks code against `rag` index of real symbols before execution |
| F3 | **Recursive reflection loops** — reflector keeps reflecting | Hard `max_depth` per loop + monotonic confidence requirement (each iteration must improve a measurable score) |
| F4 | **Memory explosion** — DB grows unboundedly | `memory.lifecycle` runs decay + dedup + summarize-and-compact passes; vector store has a TTL on low-utility memories |
| F5 | **Unsafe self-modification** — agent edits its own prompts/code | A strict allowlist: agents can update `lessons` and `patterns` rows, never `prompts/*.md` or source files |
| F6 | **Tool escape** — `exec` runs unbounded shell | Tools run via `tools.sandbox` (subprocess, CPU/memory/wallclock limits, no network unless allowlisted, FS chroot to a per-task workdir) |
| F7 | **Token blowout** — uncontrolled retries | `llm.token_meter` enforces per-task and per-session budgets; orchestrator aborts before exceeding |
| F8 | **Silent regressions** — fix introduces new bug | `Evaluator` always re-runs the failure repro before declaring success; lesson written either way |
| F9 | **Stale knowledge** — old patterns applied to new APIs | Memories carry `source_version`, `created_at`, `last_validated_at`; retriever penalizes stale items |
| F10 | **Concurrency races** — two agents mutate same state | Orchestrator owns the single mutable `RunState`; agents return *deltas*, never mutate directly |

These constraints drive the rest of the design.

---

## 3. High-level architecture

```
                            ┌──────────────────────────────────┐
                            │            Frontend              │
                            │  React + TS + Vite (chat, trace, │
                            │  memory inspector, metrics)      │
                            └───────────────┬──────────────────┘
                                            │  HTTP + SSE
                            ┌───────────────▼──────────────────┐
                            │          FastAPI Gateway         │
                            │  /chat /projects /memory /tools  │
                            │  /agents /traces /health         │
                            └───────────────┬──────────────────┘
                                            │
        ┌───────────────────────────────────┼────────────────────────────────────┐
        │                                   │                                    │
┌───────▼────────┐               ┌──────────▼──────────┐               ┌─────────▼────────┐
│  Orchestrator  │◀────traces────│  Observability bus  │───metrics────▶│   Metrics store  │
│  (state machine│               │ (structured events) │               │   (Postgres)     │
│   + router)    │               └─────────────────────┘               └──────────────────┘
└───┬────────┬───┘
    │        │
    │        │  task graph
    │        ▼
    │   ┌─────────────────────────────────────────────────────────────┐
    │   │                       Agent layer                           │
    │   │  Planner · Decomposer · ContextBuilder · Researcher ·       │
    │   │  Coder · Debugger · ToolExecutor · Evaluator · Reflector ·  │
    │   │  MemoryAgent                                                │
    │   └────────────┬───────────────┬───────────────────┬────────────┘
    │                │               │                   │
    │                │               │                   │
    │        ┌───────▼──────┐  ┌─────▼──────┐    ┌───────▼────────┐
    │        │  LLM provider│  │   Tools    │    │   Memory       │
    │        │  (OpenAI /   │  │ (sandbox,  │    │  short / long /│
    │        │   Anthropic /│  │  exec, fs, │    │  episodic /    │
    │        │   local)     │  │  search…)  │    │  semantic)     │
    │        └──────────────┘  └────────────┘    └───────┬────────┘
    │                                                    │
    │                                  ┌─────────────────┼──────────────┐
    │                                  │                 │              │
    │                            ┌─────▼─────┐   ┌───────▼──────┐ ┌─────▼────┐
    │                            │ Postgres  │   │   Qdrant     │ │  Redis   │
    │                            │ (relational│  │ (vectors)    │ │ (cache,  │
    │                            │  + JSON)   │  │              │ │  queues) │
    │                            └────────────┘  └──────────────┘ └──────────┘
    │
    └──── Safety layer wraps every external action: validators · guards · rollback
```

---

## 4. Data flow: a single user request

A turn through the system, end-to-end:

1. **Request ingress** (`api/routes/chat.py`)
   The user sends a message tied to a `session_id` (and optional `project_id`).
   The route creates a `RunContext` (request-scoped) and hands it to the orchestrator.
2. **Routing** (`orchestration/router.py`)
   The router classifies the request (Q&A · code-write · debug · research · refactor)
   using a tiny LLM call + heuristics. Output: an initial `TaskGraph`.
3. **Planning** (`agents/planner.py`)
   The Planner expands the graph into ordered subtasks. Each subtask declares its
   *required tools*, *expected outputs*, and a *success predicate*.
4. **Context building** (`agents/context_builder.py`)
   For each subtask, the ContextBuilder pulls:
   - short-term: last N turns from Redis
   - long-term: top-k memories (vector + BM25 hybrid) from Qdrant + Postgres
   - episodic: similar past task outcomes from `episodes`
   - project: indexed code chunks if a `project_id` is bound
   The result is a *budgeted* context object — every section has a max token cap.
5. **Specialist agent runs** (Coder / Debugger / Researcher)
   The agent receives the budgeted context, calls the LLM, and produces a structured
   `AgentOutput`. Outputs that propose tool execution flow into `ToolExecutor`.
6. **Tool execution** (`agents/tool_executor.py` + `tools/*`)
   Tools run inside `tools.sandbox` with explicit timeouts, FS scope, and resource
   caps. Every run yields a `ToolRun` record (stdout, stderr, exit, duration).
7. **Evaluation** (`agents/evaluator.py`)
   The Evaluator checks the subtask against its success predicate (tests pass,
   compilation succeeds, expected file exists, etc.). Produces a confidence score.
8. **Reflection** (`agents/reflector.py`)
   On failure *or* on low confidence the Reflector runs a structured root-cause
   analysis and proposes a strategy delta. The orchestrator either retries with the
   delta or escalates.
9. **Learning extraction** (`learning/reflection_engine.py` + `MemoryAgent`)
   Whatever the outcome, a `Lesson` is written: what was tried, what worked, what
   didn't, and a generalized rule. Lessons get embedded and indexed.
10. **Response synthesis**
    The orchestrator streams the final answer back over SSE while emitting trace
    events to the observability bus.

Every step above writes to traces, so a single `trace_id` reconstructs the full run
in the frontend's *Agent Trace* view.

---

## 5. Agent layer

### 5.1 Why multi-agent

A single monolithic prompt becomes incoherent past ~3 responsibilities. We split by
*responsibility* and *prompt budget*. Each agent has:

- a narrow contract (`agents/base.py::BaseAgent`)
- its own system prompt (kept short, role-specific)
- its own context budget (set by `ContextBuilder`)
- its own output schema (`schemas/agent_io.py`)

### 5.2 Roster

| Agent | Owns | Outputs |
|---|---|---|
| `Planner` | High-level decomposition | Ordered `TaskGraph` |
| `TaskDecomposer` | Splits a single task into atomic subtasks | List of `Subtask` |
| `ContextBuilder` | Builds budgeted context per subtask | `BudgetedContext` |
| `Researcher` | Pulls docs & external knowledge via RAG / web | `ResearchBundle` |
| `Coder` | Writes / edits code | `CodeChange[]` |
| `Debugger` | Reads stack traces, proposes fixes | `Patch` + `Hypothesis` |
| `ToolExecutor` | Runs tools the upstream agent requested | `ToolRun[]` |
| `Evaluator` | Verifies subtask success predicate | `EvaluationResult` |
| `Reflector` | Diagnoses failures, proposes strategy delta | `Reflection` |
| `MemoryAgent` | Decides what to remember & at what granularity | `MemoryWrite[]` |

### 5.3 Communication

Agents do **not** talk to each other directly. They return immutable
`AgentOutput`s to the orchestrator, which is the sole writer of the run state.
This kills entire classes of races and makes traces linear.

---

## 6. Memory architecture

Memory is the heart of self-learning. Four cooperating layers, each with its own
lifecycle.

### 6.1 Short-Term (volatile)

- **Where**: Redis, keyed by `session_id`.
- **What**: rolling window of last K turns, scratchpad for the current run.
- **Lifecycle**: TTL'd. On session end, summarized into long-term and dropped.

### 6.2 Long-Term (durable, semantic)

- **Where**: Postgres (canonical row + metadata) + Qdrant (vector index).
- **What**: distilled facts, user preferences, project conventions, learned
  coding patterns, failure rules.
- **Lifecycle**: written by `MemoryAgent`, deduplicated, decayed, periodically
  *consolidated* (multiple similar memories → one canonical memory).

### 6.3 Episodic (task outcomes)

- **Where**: Postgres, with vector index on `summary_embedding`.
- **What**: one row per non-trivial task — input, plan, actions, outcome,
  evaluation score, tokens spent, time. The system's *autobiography*.
- **Used for**: "have I seen something like this before, and what worked?"

### 6.4 Semantic / RAG (knowledge)

- **Where**: Qdrant.
- **What**: ingested documentation, framework patterns, internal lessons.
- **Used for**: grounding code generation in real APIs (cuts hallucinations).

### 6.5 Lifecycle policies

Three background passes (`memory/lifecycle.py`):

1. **Dedup**: when cosine similarity > 0.95 and metadata compatible, merge.
2. **Decay**: utility score = `recency * access_count * success_weight`.
   Items below threshold get summarized into a parent; raw rows archived.
3. **Compact**: chains of episodes with the same signature get rolled into a
   single "pattern" memory with a count and example list.

---

## 7. Self-improvement engine

The system improves along three axes.

### 7.1 Lessons (textual rules)

When a task fails-then-succeeds, the Reflector produces a `Lesson`:
*"When `pytest` reports `ImportError` and project uses `src/` layout, ensure
`pyproject.toml` has `[tool.pytest.ini_options] pythonpath = ['src']`."*

Lessons are embedded and pulled by the ContextBuilder for similar future tasks.

### 7.2 Patterns (parameterized templates)

Repeated *successful* subtasks are abstracted into `CodingPattern` rows:
trigger conditions, template steps, validation steps. The Planner consults the
pattern store before falling back to free-form planning.

### 7.3 Confidence calibration

Every Evaluator output carries a confidence in `[0, 1]`. We log
`(confidence, actual_outcome)` pairs and recompute a calibration curve
(`learning/confidence.py`). Future confidence values are passed through this
calibration before gating execution.

> The system never updates its own *prompts* — that path is too easy to
> destabilize. It updates the *retrievable knowledge* the prompts pull from.
> This is the safety boundary in F5.

---

## 8. RAG pipeline

```
docs ──► Loader ──► Chunker (lang-aware) ──► Embedder ──► Qdrant
                                                            │
query ──► Embedder ──┐                                       │
                     ├──► Hybrid search (vec + BM25) ──► Reranker ──► top-k
project_id, lang ────┘
```

- **Language-aware chunking**: Python uses AST boundaries; JS/TS uses tree-sitter
  via a lightweight parser; Markdown uses heading hierarchy. Falls back to
  windowed chunking only when none apply.
- **Hybrid retrieval**: vector recall + BM25 keyword recall, then a cross-encoder
  reranker. BM25 catches exact symbols; vectors catch concepts.
- **Source provenance**: every chunk records `source_uri`, `source_version`,
  `chunk_kind` (function | class | doc-section | example), enabling stale-content
  penalties and citation in answers.

---

## 9. Tool framework & sandboxing

Tools live behind a `BaseTool` interface and self-register into a `ToolRegistry`.
The `ToolExecutor` agent is the *only* path to side effects.

### 9.1 Sandbox guarantees

`tools/sandbox.py` runs each invocation as a subprocess with:

- `RLIMIT_CPU`, `RLIMIT_AS` (memory), `RLIMIT_NOFILE`
- wallclock timeout via `asyncio.wait_for`
- working directory: a per-task tempdir, the only writable path
- environment scrubbed to an allowlist
- network disabled by default; tools that need it (e.g. `web_search`) declare it

### 9.2 Built-in tools

`code_exec` · `file_ops` · `shell` · `web_search` · `docs_lookup` · `repo_analyzer`
· `pytest_runner`. Each declares: capabilities, required permissions, JSON-schema
input/output. The Planner can only emit calls to declared tools.

---

## 10. Project understanding engine

For repository-aware tasks (`project_id` bound):

1. **Indexer** walks the tree, respects `.gitignore`, language-detects each file.
2. **AST parser** extracts symbols (functions, classes, imports) and edges
   (calls, imports). Stored in Postgres as a graph (`project_symbols`,
   `project_edges`).
3. **Architecture map** clusters files by import-graph community detection,
   producing a high-level "module map" used by the Planner for *where to put*
   new code.
4. **Semantic project search** combines symbol-name search, content embedding
   search, and graph neighborhood expansion ("find this and its callers").

The agents never load the whole repo. They request *slices* of the index.

---

## 11. Safety & stability layer

`backend/app/safety/` wraps:

- **Input validators**: schema-enforce every agent output before it's executed.
- **Output validators**: lint generated code, run static checks (`ruff`, `mypy`
  in lenient mode) before saving.
- **Hallucination guard**: extracts API references from generated code and
  verifies them against the project index + RAG symbol set; flags unknowns.
- **Execution guards**: per-run depth + token + wallclock budget; circuit
  breaker stops orchestration on repeated failure.
- **Rollback**: every file write is staged in a per-run shadow dir; a successful
  evaluation promotes it, a failure discards it.
- **Integrity**: memory writes are content-addressed (sha256 of canonical form);
  duplicates short-circuit; tampering is detected on read.

---

## 12. Observability

Every component emits structured events to a single async bus (`observability/`):

```python
{ "trace_id", "span_id", "parent_span_id", "kind",
  "agent" | "tool" | "memory" | "llm",
  "phase" ("start" | "end" | "error"),
  "ts", "duration_ms", "tokens_in", "tokens_out",
  "cost_usd", "payload" }
```

Events stream to:
- Postgres (`traces` table) for queryable history
- the SSE channel of the originating request, so the frontend renders the
  trace live as it happens

This is the single most useful thing you can build for an agentic system.

---

## 13. Scalability considerations

| Concern | Approach |
|---|---|
| LLM throughput | Provider-level concurrency limits + Redis-based token-bucket; orchestrator awaits a slot before calling |
| Vector store growth | Tier hot/cold collections; cold collections compress chunks and lower vector dim |
| Postgres growth | Partition `traces` by month; archive cold partitions; `episodes` summarized into patterns |
| Hot session cache | Redis with LRU + per-session TTL |
| Long-running tasks | Background worker via `asyncio.TaskGroup`; resumable on restart via `RunState` checkpoint |
| Multi-tenant isolation | All memory/RAG queries scoped by `(user_id, project_id)`; Qdrant collection-per-tenant for sensitive cases |

---

## 14. Tradeoffs explicitly accepted

- **Latency vs quality**: we run small classifier LLM calls (Planner/Router) on a
  cheap model and large generation calls on a stronger model. Tradeoff: ~2 extra
  small calls per turn.
- **Memory recall precision vs cost**: hybrid retrieval + reranker costs more than
  pure vector search but cuts hallucinations measurably.
- **Strict sandboxing vs developer flexibility**: tools cannot escape the workdir.
  Real "plug into my repo" workflows go through the project indexer instead.
- **No self-modification of prompts**: the system improves more slowly than a
  weights-update system, but is dramatically harder to destabilize.

---

## 15. Future extensibility (built into the seams)

- **New agent**: implement `BaseAgent`, register in `agents/__init__.py`, add a
  routing rule. No orchestrator changes required.
- **New tool**: implement `BaseTool`, register, declare schema. Sandboxed by
  default.
- **New memory backend**: implement `MemoryStore` protocol; wire into
  `memory/__init__.py`.
- **New LLM provider**: implement `LLMProvider`; selected via `LLM_PROVIDER` env.
- **Fine-tuning hook**: the `episodes` + `lessons` tables are export-ready as a
  preference dataset — when fine-tuning is desired later, it's a SQL query away.

---

## 16. Repo layout (mirrors this document)

```
SelfLearing_Coding_AI_System/
├── ARCHITECTURE.md            ← you are here
├── README.md
├── docker-compose.yml
├── .env.example
├── Makefile
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── api/routes/        ← § 4
│       ├── orchestration/     ← § 4, § 10
│       ├── agents/            ← § 5
│       ├── memory/            ← § 6
│       ├── learning/          ← § 7
│       ├── rag/               ← § 8
│       ├── tools/             ← § 9
│       ├── project/           ← § 10
│       ├── safety/            ← § 11
│       ├── observability/     ← § 12
│       ├── llm/               ← provider abstraction
│       ├── db/                ← models, session
│       └── schemas/           ← pydantic I/O
├── frontend/
│   ├── Dockerfile
│   └── src/                   ← React + TS
├── infra/
│   └── postgres/init.sql
└── tests/
```

Read this document, then read the code top-down: `main.py` → `orchestration/orchestrator.py`
→ `agents/*` → `memory/*`. Everything else is leaf detail.
