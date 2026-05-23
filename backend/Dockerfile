# syntax=docker/dockerfile:1.7
# ───── Self-Learning Coding AI · backend image ─────
# Multi-stage to keep the runtime image lean.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for any C extensions (e.g. tiktoken, asyncpg fallback paths).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install deps first for layer caching: pyproject changes trigger reinstall;
# source-only edits do not.
COPY pyproject.toml ./
RUN pip install --upgrade pip wheel \
 && pip install ".[openai,anthropic]"

COPY app ./app

# ── runtime stage ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

# Minimal runtime libs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root execution user. Sandbox FS isolation still relies on rlimits +
# workdir scoping — running as a non-privileged user is defense in depth.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 slcai \
 && mkdir -p /tmp/slcai_sandbox \
 && chown -R slcai:slcai /tmp/slcai_sandbox

# Copy site-packages + the app from the builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/app /app/app

WORKDIR /app
USER slcai

EXPOSE 8000

# A real PID-1; uvicorn handles SIGTERM cleanly.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
