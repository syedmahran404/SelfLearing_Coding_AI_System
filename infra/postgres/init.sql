-- Bootstrap SQL — runs once on a fresh Postgres data dir.
-- Idempotent on every clause so re-runs are safe.

CREATE EXTENSION IF NOT EXISTS pgcrypto;     -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- trigram fuzzy search on names
CREATE EXTENSION IF NOT EXISTS btree_gin;     -- composite (jsonb + scalar) GIN

-- A small system bookkeeping table used by app.db.migrate
CREATE TABLE IF NOT EXISTS slcai_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT INTO slcai_meta (key, value)
VALUES ('schema_initialized', NOW()::text)
ON CONFLICT (key) DO NOTHING;
