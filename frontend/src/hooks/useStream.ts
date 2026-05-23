import { useCallback, useEffect, useRef, useState } from "react";
import { ChatEvent, ChatStreamHandle, streamChat } from "../api/client";

export interface ChatRunState {
  events: ChatEvent[];
  status: string;
  answer: string;
  trace_id?: string;
  inFlight: boolean;
  error?: string;
}

const initial: ChatRunState = {
  events: [],
  status: "idle",
  answer: "",
  inFlight: false,
};

/**
 * useChatStream — drives one chat run end-to-end.
 *
 * Aggregates `answer` chunks, tracks the latest `status`, and exposes the
 * full event stream so the AgentTrace view can render per-step detail.
 */
export function useChatStream() {
  const [state, setState] = useState<ChatRunState>(initial);
  const handleRef = useRef<ChatStreamHandle | null>(null);

  useEffect(
    () => () => {
      handleRef.current?.close();
    },
    [],
  );

  const send = useCallback(
    (message: string, opts?: { session_id?: string; project_id?: string }) => {
      handleRef.current?.close();
      setState({ ...initial, inFlight: true, status: "starting" });

      const handle = streamChat({
        message,
        session_id: opts?.session_id,
        project_id: opts?.project_id,
        onEvent: (ev) =>
          setState((s) => {
            const next: ChatRunState = {
              ...s,
              events: [...s.events, ev],
              trace_id: s.trace_id ?? handle.trace_id,
            };
            const data = ev.data as Record<string, unknown> | undefined;
            if (ev.type === "status" && data && typeof data.status === "string") {
              next.status = data.status;
            }
            if (ev.type === "answer" && data && typeof data.answer === "string") {
              next.answer = data.answer;
            }
            if (ev.type === "done") {
              next.inFlight = false;
              next.status = (data?.status as string) ?? "done";
            }
            if (ev.type === "error") {
              next.inFlight = false;
              next.status = "error";
              next.error =
                (data && typeof data === "object" && "message" in data
                  ? String((data as { message: string }).message)
                  : "unknown error");
            }
            return next;
          }),
        onError: (err) =>
          setState((s) => ({
            ...s,
            inFlight: false,
            status: "error",
            error: String((err as Error)?.message ?? err),
          })),
        onDone: () =>
          setState((s) => (s.inFlight ? { ...s, inFlight: false, status: "done" } : s)),
      });

      handleRef.current = handle;
    },
    [],
  );

  const cancel = useCallback(() => {
    handleRef.current?.close();
    handleRef.current = null;
    setState((s) => ({ ...s, inFlight: false, status: "cancelled" }));
  }, []);

  return { state, send, cancel };
}
