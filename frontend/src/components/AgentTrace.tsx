import { useEffect, useState } from "react";
import { TraceDetail, TraceEvent, api } from "../api/client";

/** AgentTrace — fetches a persisted trace by id and renders a span tree. */
export function AgentTrace({ traceId }: { traceId?: string }) {
  const [trace, setTrace] = useState<TraceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!traceId) {
      setTrace(null);
      return;
    }
    let alive = true;
    api
      .trace(traceId)
      .then((d) => alive && setTrace(d))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [traceId]);

  if (!traceId) {
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Agent Trace</h2>
        </header>
        <p className="muted">No active trace. Send a chat to see one here.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Agent Trace</h2>
        <code className="dim">{traceId.slice(0, 12)}…</code>
      </header>
      {error && <p className="error">{error}</p>}
      {!trace ? (
        <p className="muted">loading…</p>
      ) : (
        <SpanTree events={trace.events} />
      )}
    </section>
  );
}

function SpanTree({ events }: { events: TraceEvent[] }) {
  // Keep only END events (those carry duration + tokens) for the tree;
  // START events are implied. ERROR events bubble up via styling.
  const ends = events.filter((e) => e.phase !== "start");
  const byParent = new Map<string | null, TraceEvent[]>();
  for (const ev of ends) {
    const k = ev.parent_span_id ?? null;
    const list = byParent.get(k) ?? [];
    list.push(ev);
    byParent.set(k, list);
  }
  // Sort siblings by ts.
  for (const arr of byParent.values()) arr.sort((a, b) => a.ts - b.ts);

  const renderChildren = (parent: string | null, depth: number) => {
    const children = byParent.get(parent) ?? [];
    return children.map((ev) => (
      <li
        key={ev.span_id}
        className={`span span-${ev.phase} kind-${ev.kind}`}
        style={{ marginLeft: depth * 14 }}
      >
        <span className="span-kind">{ev.kind}</span>
        <span className="span-name">{ev.name}</span>
        {ev.duration_ms != null && (
          <span className="span-dur">{Math.round(ev.duration_ms)}ms</span>
        )}
        {ev.tokens_in != null && (ev.tokens_in > 0 || (ev.tokens_out ?? 0) > 0) && (
          <span className="span-tok">
            {ev.tokens_in}/{ev.tokens_out ?? 0}t
          </span>
        )}
        {ev.error && <span className="error">{ev.error}</span>}
        <ul className="trace-children">{renderChildren(ev.span_id, depth + 1)}</ul>
      </li>
    ));
  };

  return <ul className="trace-tree">{renderChildren(null, 0)}</ul>;
}
