/**
 * API client.
 *
 * Reads `VITE_API_BASE_URL` so the same build can target dev/staging/prod.
 * Falls back to `/api` (the dev-server proxy in `vite.config.ts`).
 */

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api";

export class ApiError extends Error {
  status: number;
  body?: unknown;
  constructor(message: string, status: number, body?: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const headers = new Headers(init.headers ?? {});
  if (!headers.has("content-type") && init.body) {
    headers.set("content-type", "application/json");
  }
  const r = await fetch(url, { ...init, headers });
  const text = await r.text();
  let body: unknown = undefined;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!r.ok) {
    throw new ApiError(`${r.status} ${r.statusText}`, r.status, body);
  }
  return body as T;
}

// ── domain helpers ──

export interface AgentDescriptor {
  name: string;
  responsibility: string;
}
export const api = {
  health: () => request<{ status: string }>("/health"),
  ready: () => request<{ status: string; checks: Record<string, string> }>("/ready"),
  agents: () =>
    request<{ agents: AgentDescriptor[] }>("/agents").then((d) => d.agents),
  metrics: () =>
    request<{
      uptime_s: number;
      totals: Record<string, number>;
      by_kind: Record<string, Record<string, number>>;
      by_name: Record<string, Record<string, number>>;
    }>("/metrics"),
  recentTraces: () => request<{ traces: TraceSummary[] }>("/traces/recent"),
  trace: (id: string) => request<TraceDetail>(`/traces/${id}`),
  // memory
  memorySearch: (query: string, top_k = 8, kind?: string) =>
    request<MemoryRow[]>("/memory/search", {
      method: "POST",
      body: JSON.stringify({ query, top_k, kind }),
    }),
  memoryList: (kind?: string) =>
    request<MemoryRow[]>(`/memory${kind ? `?kind=${encodeURIComponent(kind)}` : ""}`),
  // projects
  listProjects: () => request<ProjectRow[]>("/projects"),
  createProject: (body: ProjectCreate) =>
    request<ProjectRow>("/projects", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // tools
  listTools: () => request<{ tools: ToolDescriptor[] }>("/tools"),
};

// ── types mirroring the backend's Pydantic schemas ──

export interface MemoryRow {
  id: string;
  kind: string;
  content: string;
  summary: string | null;
  tags: string[];
  confidence: number;
  utility: number;
  access_count: number;
  success_count: number;
  failure_count: number;
  created_at: string;
  last_accessed_at: string;
  score?: number | null;
}

export interface ProjectRow {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  languages: string[];
  repo_root: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectCreate {
  name: string;
  slug: string;
  description?: string;
  languages?: string[];
  repo_root?: string;
}

export interface ToolDescriptor {
  name: string;
  description: string;
  permissions: string[];
  default_timeout_s: number;
  safe_default: boolean;
}

export interface TraceSummary {
  trace_id: string;
  spans: number;
  errors: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  kinds: string[];
  duration_ms: number;
  ts_min: number;
  ts_max: number;
}

export interface TraceEvent {
  span_id: string;
  parent_span_id: string | null;
  kind: string;
  name: string;
  phase: "start" | "end" | "error" | "event";
  ts: number;
  duration_ms: number | null;
  tokens_in: number | null;
  tokens_out: number | null;
  cost_usd: number | null;
  payload: Record<string, unknown>;
  error: string | null;
}

export interface TraceDetail {
  trace_id: string;
  events: TraceEvent[];
}

// ── chat: SSE streaming ──

export interface ChatEvent {
  type: string;
  data: unknown;
}

export interface ChatStreamHandle {
  close: () => void;
  trace_id: string;
}

/**
 * Open a streaming chat run. The server returns SSE chunks tagged by `event`.
 * We use `fetch` + a manual stream reader so we can POST a body (EventSource
 * is GET-only and doesn't accept request bodies).
 */
export function streamChat(opts: {
  message: string;
  session_id?: string;
  project_id?: string;
  user_id?: string;
  trace_id?: string;
  onEvent: (ev: ChatEvent) => void;
  onError?: (err: unknown) => void;
  onDone?: () => void;
}): ChatStreamHandle {
  const controller = new AbortController();
  const trace_id = opts.trace_id ?? crypto.randomUUID().replace(/-/g, "");
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "x-trace-id": trace_id,
    accept: "text/event-stream",
  };

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          message: opts.message,
          session_id: opts.session_id,
          project_id: opts.project_id,
          user_id: opts.user_id,
          stream: true,
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        throw new ApiError(`${res.status} ${res.statusText}`, res.status);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by blank lines.
        let idx: number;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const raw = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const ev = parseSseBlock(raw);
          if (ev) opts.onEvent(ev);
        }
      }
      opts.onDone?.();
    } catch (e) {
      if ((e as DOMException)?.name !== "AbortError") {
        opts.onError?.(e);
      }
    }
  })();

  return {
    trace_id,
    close: () => controller.abort(),
  };
}

function parseSseBlock(block: string): ChatEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (!line) continue;
    if (line.startsWith(":")) continue; // comment
    const colon = line.indexOf(":");
    const field = colon >= 0 ? line.slice(0, colon) : line;
    const value = colon >= 0 ? line.slice(colon + 1).trimStart() : "";
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  const dataStr = dataLines.join("\n");
  let data: unknown = dataStr;
  try {
    data = JSON.parse(dataStr);
  } catch {
    /* leave as string */
  }
  return { type: event, data };
}
