import { FormEvent, useState } from "react";
import { useChatStream } from "../hooks/useStream";

interface Props {
  sessionId?: string;
  projectId?: string;
  onTraceId?: (id: string) => void;
}

/** ChatPanel — text input + streamed answer + agent activity list. */
export function ChatPanel({ sessionId, projectId, onTraceId }: Props) {
  const { state, send, cancel } = useChatStream();
  const [text, setText] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim() || state.inFlight) return;
    send(text.trim(), { session_id: sessionId, project_id: projectId });
    setText("");
  };

  // Surface the trace_id once we have it.
  if (state.trace_id && onTraceId) onTraceId(state.trace_id);

  return (
    <section className="panel chat">
      <header className="panel-head">
        <h2>Chat</h2>
        <span className={`pill pill-${statusKind(state.status)}`}>
          {state.status}
        </span>
      </header>

      <div className="chat-body">
        <div className="chat-answer">
          {state.answer ? (
            <pre className="answer-text">{state.answer}</pre>
          ) : state.inFlight ? (
            <p className="muted">…thinking…</p>
          ) : (
            <p className="muted">Ask a question or describe what to build.</p>
          )}
          {state.error && <p className="error">{state.error}</p>}
        </div>

        <ActivityList events={state.events} />
      </div>

      <form onSubmit={submit} className="chat-input">
        <textarea
          rows={3}
          placeholder="e.g. write a Python function that does X"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={state.inFlight}
        />
        <div className="chat-actions">
          <button type="submit" disabled={state.inFlight || !text.trim()}>
            Send
          </button>
          {state.inFlight && (
            <button type="button" onClick={cancel} className="ghost">
              Cancel
            </button>
          )}
        </div>
      </form>
    </section>
  );
}

function ActivityList({ events }: { events: { type: string; data: unknown }[] }) {
  if (events.length === 0) return null;
  return (
    <ul className="activity">
      {events
        .filter((e) =>
          ["status", "plan", "subtask", "tool", "evaluation", "reflection", "memory"].includes(
            e.type,
          ),
        )
        .map((e, i) => (
          <li key={i} className={`act act-${e.type}`}>
            <span className="act-tag">{e.type}</span>
            <code>{summarize(e)}</code>
          </li>
        ))}
    </ul>
  );
}

function summarize(e: { type: string; data: unknown }): string {
  const d = (e.data ?? {}) as Record<string, unknown>;
  if (e.type === "status") return `→ ${d.status}`;
  if (e.type === "plan") return `${(d.subtasks as unknown[] | undefined)?.length ?? 0} subtask(s): ${d.title ?? ""}`;
  if (e.type === "subtask") return `${d.title ?? d.id ?? ""} [${d.status ?? "?"}] (attempt ${d.attempt ?? d.attempts ?? "?"})`;
  if (e.type === "tool") return `${d.tool} → ${d.ok ? "ok" : "fail"}`;
  if (e.type === "evaluation") return `passed=${d.passed} score=${d.score}`;
  if (e.type === "reflection") return `${(d.root_cause as string | undefined) ?? "(root cause)"}`;
  if (e.type === "memory") return `mem ${d.kind}: ${(d.content as string | undefined)?.slice(0, 80) ?? ""}`;
  return JSON.stringify(d).slice(0, 120);
}

function statusKind(s: string): string {
  if (s === "success" || s === "done") return "ok";
  if (s === "error" || s === "failed") return "err";
  if (s === "running" || s === "planning" || s === "reflecting" || s === "starting")
    return "warn";
  return "neutral";
}
