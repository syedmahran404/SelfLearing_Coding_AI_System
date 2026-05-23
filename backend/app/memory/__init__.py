"""Memory subsystem.

Four cooperating layers (see ARCHITECTURE.md §6):

- short_term  : Redis rolling window per session
- long_term   : Postgres canonical row + Qdrant vector
- episodic    : task outcomes — the system's autobiography
- compression : summarization / consolidation passes
- lifecycle   : background dedup / decay / prune

`MemoryService` is the public facade — agents and the orchestrator only
import the service.
"""

from app.memory.lifecycle import LifecycleWorker
from app.memory.service import MemoryService

__all__ = ["MemoryService", "LifecycleWorker"]
