# Self-Learning Coding AI

A production-oriented autonomous coding system that **plans, codes, debugs,
reflects, and learns from its own outcomes**. Multi-agent orchestration on top
of a layered memory system (short-term / long-term / episodic / semantic) with a
sandboxed tool framework, RAG-grounded code generation, and a reflection-driven
self-improvement loop.

> Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) first. It is the design contract
> the code follows. This README is just how to run it.

---

## What it does

- **Plans** a coding task into ordered, verifiable subtasks.
- **Builds context** by retrieving relevant memories, lessons, and code chunks
  (hybrid vector + BM25 + reranker) within strict token budgets.
- **Writes code** via specialist agents (Coder, Debugger, Researcher) that run
  inside a sandboxed tool framework.
- **Evaluates** every output against an explicit success predicate (tests pass,
  build succeeds, file exists, etc.).
- **Reflects** on failures, extracts a generalized lesson, and persists it to
  long-term memory.
- **Learns** by accumulating lessons, patterns, and episodes — so the same class
  of task gets cheaper and more reliable over time.

The system never modifies its own prompts or source. Improvement is purely
*through retrievable memory*, which is the safety boundary.

---

## Stack

| Layer | Tech |
|---|---|
| Backend | Python 3.11, FastAPI, async SQLAlchemy, Pydantic v2 |
| LLM | Provider-agnostic (OpenAI / Anthropic / local), streaming |
| Relational store | PostgreSQL 16 |
| Vector store | Qdrant |
| Cache / queues | Redis 7 |
| Frontend | React 18 + TypeScript + Vite |
| Infra | Docker Compose |
| Tests | pytest, pytest-asyncio |

---

## Quickstart (Docker)

```bash
git clone <this-repo>
cd SelfLearing_Coding_AI_System
cp .env.example .env
# edit .env: set OPENAI_API_KEY (or ANTHROPIC_API_KEY, or LLM_PROVIDER=local)

make up
make logs
```

Endpoints:

- API:        http://localhost:8000
- API docs:   http://localhost:8000/docs
- Frontend:   http://localhost:5173
- Qdrant:     http://localhost:6333/dashboard

Health: `curl http://localhost:8000/health`

---

## Quickstart (local, without Docker)

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# requires running postgres, redis, qdrant locally — or run them via:
#   docker compose up -d postgres redis qdrant
make dev      # uvicorn with hot reload

# frontend (separate shell)
cd frontend
npm install
npm run dev
```

---

## Project layout

```
.
├── ARCHITECTURE.md          # design doc — read first
├── README.md
├── docker-compose.yml
├── .env.example
├── Makefile
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── app/
│       ├── main.py                # FastAPI app
│       ├── config.py              # settings
│       ├── api/routes/            # HTTP routes
│       ├── orchestration/         # state machine, router, run state
│       ├── agents/                # planner, coder, debugger, ...
│       ├── memory/                # short / long / episodic / semantic
│       ├── learning/              # reflection, evaluation, patterns
│       ├── rag/                   # ingestion, chunking, retrieval
│       ├── tools/                 # sandboxed tools
│       ├── project/               # repo indexer, dep graph
│       ├── safety/                # validators, guards, rollback
│       ├── observability/         # tracing, metrics, logger
│       ├── llm/                   # provider abstraction
│       ├── db/                    # models, session
│       └── schemas/               # pydantic I/O
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   └── src/
└── infra/
    └── postgres/init.sql
```

---

## Running tests

```bash
make test                   # backend pytest
cd backend && pytest -q     # equivalent
```

Tests are organized by layer (`tests/test_memory.py`, `tests/test_tools.py`,
`tests/test_agents.py`, `tests/test_api.py`). The agent and tool tests use a
deterministic `LocalProvider` LLM stub so they run without API keys.

---

## Configuration cheatsheet

Everything is in `.env` (see `.env.example`):

- `LLM_PROVIDER` — `openai` | `anthropic` | `local` (deterministic stub)
- `LLM_DEFAULT_MODEL` — global default
- `LLM_PLANNER_MODEL` — usually a small/cheap model
- `LLM_CODER_MODEL` — your strongest model
- `SANDBOX_*` — resource limits for tool execution
- `SAFETY_*` — recursion depth, confidence floor, hallucination block
- `MEMORY_*` — dedup threshold, decay halflife, lifecycle interval

---

## Development workflow

```bash
make lint          # ruff + eslint
make format        # ruff format + prettier
make typecheck     # mypy + tsc
make migrate       # apply DB migrations
```

The repo is committed to:

- **strict typing** in the backend (`mypy`)
- **structured logging** — every event carries a `trace_id`
- **dependency injection** — agents/tools are constructed by `app.main` and
  passed down; no global state outside of explicit registries

---

## License

MIT
