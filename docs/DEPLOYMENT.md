# Deployment

This document covers running the system in production-grade settings.
For local development, see [`README.md`](../README.md).

---

## 1. Topology

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Client     │ ─► │  reverse     │ ─► │   backend    │
│  (browser)   │    │   proxy      │    │  (uvicorn)   │
└──────────────┘    │  (nginx /    │    │  N replicas  │
                    │   ALB / GCS) │    └─────┬────────┘
                    └──────┬───────┘          │
                           │                  ▼
                           │           ┌──────────────┐
                           ▼           │  postgres    │
                    ┌──────────────┐   │  redis       │
                    │   frontend   │   │  qdrant      │
                    │  (static)    │   └──────────────┘
                    └──────────────┘
```

The backend is **stateless across replicas** — the only stateful stores
are Postgres, Redis, and Qdrant. You can run any number of backend
replicas behind a load balancer.

In-memory state that is *not* shared across replicas:

- `MetricsCollector` (per-process counter snapshot — that's by design)
- `CircuitBreaker` (per-process — see §6 if you want process-shared)
- `EmbeddingService` LRU cache
- The `Tracer`'s per-trace SSE queues are bound to the replica that
  served the request; SSE clients should target the same replica via a
  sticky session or the `x-trace-id` header.

---

## 2. Required services

| Service     | Min version | Why                              |
|-------------|-------------|----------------------------------|
| PostgreSQL  | 14+         | jsonb + GIN indexes used heavily |
| Redis       | 6+          | streams not used; any 6+ works   |
| Qdrant      | 1.9+        | payload indexes API              |
| Python      | 3.11+       | `dataclass(slots=True)`, `match` |

---

## 3. First-run sequence

```bash
# 1. Provision env
cp .env.example .env
$EDITOR .env   # set OPENAI_API_KEY (or ANTHROPIC_API_KEY), DATABASE_URL, etc.

# 2. Bring up infra
docker compose up -d postgres redis qdrant

# 3. Apply schema
make migrate            # python -m app.db.migrate (idempotent)

# 4. Bring up app
docker compose up -d backend frontend

# 5. Smoke
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
```

`/ready` returns 200 only when Postgres, Redis, and Qdrant are all
reachable. Use it as the orchestrator's readiness probe.

---

## 4. Configuration knobs that matter in production

| Env                                  | Suggested prod value                                        | Notes |
|--------------------------------------|-------------------------------------------------------------|-------|
| `LLM_DEFAULT_MODEL`                  | strongest you can afford                                    | per-agent overrides via `LLM_PLANNER_MODEL` / `LLM_CODER_MODEL` |
| `LLM_MAX_TOKENS_PER_TASK`            | 200_000                                                     | hard ceiling per chat turn |
| `LLM_MAX_TOKENS_PER_SESSION`         | 2_000_000                                                   | hard ceiling per session |
| `LLM_REQUEST_TIMEOUT_S`              | 120                                                         | honors provider's typical SLA |
| `SAFETY_MAX_RECURSION_DEPTH`         | 4                                                           | translates into orchestrator iteration cap |
| `SAFETY_MIN_CONFIDENCE_TO_EXECUTE`   | 0.55–0.7                                                    | higher → fewer tool runs, fewer side-effects |
| `SAFETY_HALLUCINATION_BLOCK`         | `true`                                                      | keep on for code-write workflows |
| `SAFETY_DRYRUN_FILE_WRITES`          | `false` in prod, `true` for evaluation runs                 | |
| `MEMORY_DECAY_HALFLIFE_DAYS`         | 30                                                          | tune per usage volume |
| `MEMORY_LIFECYCLE_INTERVAL_S`        | 3600                                                        | dedup/decay frequency |
| `OBSERVABILITY_TRACE_TO_DB`          | `true`                                                      | required for `/traces` UI |
| `SANDBOX_MAX_MEMORY_MB`              | 512–1024                                                    | tighten for shared infra |
| `SANDBOX_MAX_CPU_S`                  | 20                                                          | per tool invocation |
| `SANDBOX_NETWORK_DEFAULT`            | `deny`                                                      | flip per-tool only |

---

## 5. Resource sizing rules of thumb

- **Backend pod**: 1 vCPU + 1 GiB RAM per replica handles ~30 concurrent
  chat sessions. The bottleneck is almost always the LLM provider's
  rate limits, not the backend.
- **Postgres**: 4 GiB RAM, 50 GiB disk for ~1M episodes + 5M traces.
  Partition `traces` by month; archive cold partitions to S3.
- **Redis**: 256 MiB is plenty unless you crank `REDIS_SHORT_TERM_TTL_S`.
- **Qdrant**: ~1 KiB per point at the default vector size. 10M points ≈
  10 GiB, comfortable on 16 GiB RAM.

---

## 6. Cross-process shared state (optional)

The `CircuitBreaker` is per-process today. To make it cluster-wide:

```python
# In a custom subclass:
class RedisCircuitBreaker(CircuitBreaker):
    def is_open(self, agent, intent):
        return await self._redis.exists(f"slcai:cb:{agent}:{intent}:open") == 1

    async def record_failure(self, agent, intent):
        # INCR with TTL; trip if count exceeds threshold.
        ...
```

Wire it via `Orchestrator(... circuit_breaker=RedisCircuitBreaker(...))`.
The default per-process implementation is enough for single-replica or
sticky-session deployments.

---

## 7. Backups and disaster recovery

- **Postgres**: daily logical dump (`pg_dump -Fc`). The episodic memory
  table is the AI's *autobiography* — its loss kills the system's
  acquired learning, not just session state.
- **Qdrant**: snapshot the per-collection volumes (`qdrant snapshot`).
- **Redis**: optional. Short-term memory survives in episodes; losing
  Redis only loses in-flight session windows.

---

## 8. Migration playbook

1. Take a logical backup of Postgres.
2. Take a Qdrant snapshot.
3. Deploy the new code.
4. Run `python -m app.db.migrate` (idempotent — adds columns/indexes;
   never drops).
5. Run `POST /memory/lifecycle/run` to recompute utility scores after
   any schema change that touches that path.
6. Watch `/metrics` for elevated `errors` count for ~10 minutes.
7. If degraded: roll back the deployment. The DB schema is forward
   compatible with the previous code by design (only additive
   migrations).

---

## 9. Observability in production

- **Logs**: emit JSON (set `APP_LOG_LEVEL=INFO`, `APP_DEBUG=false`).
  Every record carries `trace_id`. Ship to Loki / CloudWatch / etc.
- **Traces**: query the `traces` table by `trace_id`. The frontend
  `/traces/{id}` view is the developer ergonomics; the table itself is
  the source of truth.
- **Metrics**: `/metrics` returns the live process snapshot.
  For Prometheus, attach an exporter that reads
  `app.observability.metrics.MetricsCollector.snapshot()` periodically.

---

## 10. Security checklist

- [ ] `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` stored in a secrets manager,
      never in the image.
- [ ] DB user has `INSERT/UPDATE/DELETE` only on the app's schema (no
      DROP/ALTER outside of migrations).
- [ ] Backend container runs as **uid 10001** (the Dockerfile sets this).
- [ ] Sandbox tools cannot reach the public internet
      (`SANDBOX_NETWORK_DEFAULT=deny`); enable per tool only.
- [ ] Reverse proxy enforces TLS, HSTS, and a sane CSP.
- [ ] CORS allowlist is restricted (`APP_CORS_ORIGINS=https://your.app`).
- [ ] `/metrics` and `/memory/lifecycle/run` are protected by an
      operator-only auth path (add a FastAPI dependency with an admin
      token check).
