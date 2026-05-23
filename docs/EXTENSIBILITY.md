# Extensibility cookbook

The system is built for change at the *seams*, not by editing the core.
Each section below shows the precise extension point + an end-to-end
example. Internal types are referenced by their import path so you can
follow along in code.

---

## Add a new agent

**Seam:** `app.agents.base.BaseAgent` + `app.agents.__init__.build_agent_registry`.

```python
# app/agents/security_reviewer.py
from app.agents.base import BaseAgent
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput

SECURITY_PROMPT = """\
You are the SECURITY REVIEWER agent. Read the proposed code changes and
flag injection, unsafe deserialization, secret leakage, and unsafe shell.
Output strictly matches the schema. No prose.
"""

_SCHEMA = {
    "type": "object",
    "required": ["findings", "confidence"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["severity", "location", "issue", "fix"],
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warn", "high"]},
                    "location": {"type": "string"},
                    "issue": {"type": "string"},
                    "fix": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "number"},
    },
}


class SecurityReviewerAgent(BaseAgent):
    name = "security_reviewer"
    responsibility = "Review proposed code for common security issues."

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(self._tracer, "agent.security_reviewer.run", SpanKind.AGENT):
            data, usage = await self._llm_json(
                system=SECURITY_PROMPT,
                user=str(ai.extras.get("code_to_review", "")),
                schema=_SCHEMA,
                temperature=0.1,
                max_tokens=900,
                purpose="agent.security_reviewer",
            )
            findings = data.get("findings") or []
            return AgentOutput(
                agent=self.name,
                summary=f"{len(findings)} security finding(s).",
                confidence=float(data.get("confidence", 0.4)),
                metadata={"findings": findings},
            )
```

Then register it in `app/agents/__init__.py::build_agent_registry`:

```python
from app.agents.security_reviewer import SecurityReviewerAgent
reg.register(SecurityReviewerAgent(llm=llm, tracer=tracer, settings=settings))
```

The Planner can now route subtasks to `agent: "security_reviewer"`. No
orchestrator changes required.

---

## Add a new tool

**Seam:** `app.tools.base.BaseTool` + `app.tools.registry.build_default_registry`.

```python
# app/tools/git_status.py
from app.tools.base import BaseTool, Permission, ToolInput, ToolResult, ToolSchema
from app.tools.sandbox import Sandbox


class GitStatusTool(BaseTool):
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    schema = ToolSchema(
        name="git_status",
        description="Run `git status --porcelain` inside the workdir.",
        permissions=[Permission.EXEC, Permission.READ],
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {"clean": {"type": "boolean"}, "files": {"type": "array"}},
        },
        default_timeout_s=10,
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        sr = await self._sandbox.run(
            ["git", "status", "--porcelain"], cwd=ti.workdir, timeout_s=ti.timeout_s
        )
        files = [line[3:] for line in sr.stdout.splitlines() if line.strip()]
        return ToolResult(
            ok=sr.exit_code == 0 and not sr.timed_out,
            output={"clean": not files, "files": files},
            stdout=sr.stdout, stderr=sr.stderr,
            exit_code=sr.exit_code, duration_ms=sr.duration_ms,
        )
```

Register it in `build_default_registry`:

```python
from app.tools.git_status import GitStatusTool
reg.register(GitStatusTool(sandbox))
```

The orchestrator's `ToolExecutor` can now route any agent's
`ToolInvocation(tool="git_status", ...)` through the same sandbox +
permission + timeout pipeline as built-in tools.

---

## Add a new LLM provider

**Seam:** `app.llm.provider.LLMProvider` + `app.llm.__init__.build_llm_provider`.

Implement the three abstract methods (`complete`, `stream`, `embed`),
attach a `TokenMeter`, and add a branch in `build_llm_provider`:

```python
# app/llm/together_provider.py
from app.llm.provider import LLMProvider
# (largely identical to OpenAIProvider — Together is OpenAI-compatible)
```

```python
# app/llm/__init__.py
if provider == "together":
    from app.llm.together_provider import TogetherProvider
    return TogetherProvider(settings)
```

Then `LLM_PROVIDER=together` in `.env` and the rest of the system
doesn't notice the swap.

---

## Add a new memory backend

**Seam:** `app.memory.long_term.LongTermMemory` is the canonical write/read
contract. Two pieces to swap:

1. **Vector store** — replace `app.db.qdrant.QdrantStore` with another
   client conforming to its narrow interface (`upsert`, `delete`,
   `search`, `count`, `ensure_collections`).
2. **Relational** — `LongTermMemory` already uses SQLAlchemy through
   `AsyncSession`; swapping Postgres for SQLite/CockroachDB is a
   `DATABASE_URL` change.

If you want a second *kind* of memory (e.g. graph memory), add a new
class alongside `LongTermMemory` and expose it on `MemoryService` as a
sibling property. Don't replace the existing layers — they're consumed
by the agent layer through `MemoryService`.

---

## Add a new chunking strategy

**Seam:** `app.rag.chunking.chunk` dispatches by extension. Add a new
language extractor:

```python
# In app/rag/chunking.py
def _chunk_rust(text, source_uri):
    # extract `fn`, `struct`, `impl`, etc.
    ...

# wire it into chunk():
if lang == "rust":
    out = list(_chunk_rust(text, source_uri))
    return out or list(_chunk_window(text, source_uri, lang))
```

Extend `_LANG_BY_EXT` if needed. RAG retrieval picks up the new chunks
on the next ingestion run; no other code changes.

---

## Add a new safety check

**Seam:** `app.safety` — pure functions / classes consumed by the
orchestrator and route handlers.

A new check is just a new function:

```python
# app/safety/secrets.py
import re
_SECRET_RES = (re.compile(r"AKIA[0-9A-Z]{16}"), re.compile(r"sk-[a-zA-Z0-9]{32,}"))

def has_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_RES)
```

Then call it in `app.safety.validators.validate_agent_output` so it's
applied uniformly:

```python
for c in out.code_changes:
    if c.new_content and has_secret(c.new_content):
        raise ValidationError(f"refusing change containing apparent secret: {c.path}")
```

---

## Persist learning artifacts as a fine-tuning dataset

The `episodes` and `lessons` tables are export-ready. One SQL query +
a small script gets you a JSONL dataset:

```sql
SELECT input  AS prompt,
       summary AS completion,
       outcome,
       score,
       confidence,
       intent,
       created_at
FROM   episodes
WHERE  outcome = 'success' AND score >= 0.85
ORDER  BY created_at DESC
LIMIT  10000;
```

Pair each row with the lesson(s) in scope at the time (`created_at <
lesson.created_at` filter) and you have a preference dataset for DPO /
ORPO. The system never assumes you'll fine-tune — but the schema makes
it cheap to start.

---

## Add a new HTTP route

**Seam:** `app.api.routes.register_routes`. New module under
`app/api/routes/`, expose a `router = APIRouter(...)`. The seam does the
rest:

```python
# app/api/routes/admin.py
from fastapi import APIRouter

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/echo")
async def echo(): return {"ok": True}
```

The route loader is dynamic (`__import__` over a list of module names),
so the only change required is adding `"admin"` to that list.

---

## Add a new background job

**Seam:** anything implementing `start() / shutdown()` and registered on
the `Runtime`. Pattern: long-lived `asyncio.Task`, cooperative
cancellation, idempotent on restart. See
`app.memory.lifecycle.LifecycleWorker` for the reference shape.

Wire it in `bootstrap.build_runtime` near the existing memory worker;
add to `Runtime.shutdown` so SIGTERM cleans up.

---

## Things you should NOT extend by editing core code

- The `RunState` recursion guards (`max_iterations`,
  `insertions_remaining`, `consecutive_subtask_failures`). Tightening is
  fine via env / settings; loosening will reintroduce the recursion bug
  this fix closed.
- Direct mutation of `state.subtasks` outside of `_insert_subtasks`.
  That helper enforces the per-run insertion budget.
- Agent-to-agent direct calls. Always return `AgentOutput` to the
  orchestrator and let it route. Direct calls reintroduce the kind of
  recursion failure that motivated the safety layer in the first place.
