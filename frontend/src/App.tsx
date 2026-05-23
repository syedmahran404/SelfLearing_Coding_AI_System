import { useState } from "react";
import { AgentTrace } from "./components/AgentTrace";
import { ChatPanel } from "./components/ChatPanel";
import { MemoryInspector } from "./components/MemoryInspector";
import { MetricsPanel } from "./components/MetricsPanel";
import { ProjectExplorer } from "./components/ProjectExplorer";

type Tab = "chat" | "memory" | "metrics";

export function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [projectId, setProjectId] = useState<string | undefined>(undefined);
  const [traceId, setTraceId] = useState<string | undefined>(undefined);

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          Self-Learning Coding AI <span className="dim">— v0.1</span>
        </h1>
        <nav>
          {(["chat", "memory", "metrics"] as Tab[]).map((t) => (
            <button
              key={t}
              type="button"
              className={tab === t ? "tab active" : "tab"}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      <main className={`layout layout-${tab}`}>
        {tab === "chat" && (
          <>
            <aside className="col-left">
              <ProjectExplorer
                selectedId={projectId}
                onSelect={(id) => setProjectId(id)}
              />
            </aside>
            <section className="col-main">
              <ChatPanel
                projectId={projectId}
                onTraceId={(id) => setTraceId(id)}
              />
            </section>
            <aside className="col-right">
              <AgentTrace traceId={traceId} />
            </aside>
          </>
        )}

        {tab === "memory" && (
          <section className="col-full">
            <MemoryInspector />
          </section>
        )}

        {tab === "metrics" && (
          <section className="col-full">
            <MetricsPanel />
          </section>
        )}
      </main>
    </div>
  );
}
