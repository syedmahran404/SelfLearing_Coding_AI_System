import { FormEvent, useEffect, useState } from "react";
import { MemoryRow, api } from "../api/client";

/** MemoryInspector — search + list long-term memories with utility scores. */
export function MemoryInspector() {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<string>("");
  const [rows, setRows] = useState<MemoryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.memoryList(kind || undefined);
      setRows(data);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind]);

  const search = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) {
      load();
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.memorySearch(query.trim(), 10, kind || undefined);
      setRows(data);
    } catch (e2) {
      setError(String((e2 as Error).message ?? e2));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Memory</h2>
        <span className="dim">{rows.length} item(s)</span>
      </header>

      <form onSubmit={search} className="memory-controls">
        <input
          placeholder="search memories…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="">all kinds</option>
          <option value="preference">preference</option>
          <option value="convention">convention</option>
          <option value="failure_rule">failure_rule</option>
          <option value="success_rule">success_rule</option>
          <option value="fact">fact</option>
          <option value="lesson">lesson</option>
        </select>
        <button type="submit" disabled={loading}>
          {loading ? "…" : "Search"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      <ul className="memory-list">
        {rows.map((m) => (
          <li key={m.id} className={`memory-item kind-${m.kind}`}>
            <div className="memory-item-head">
              <span className="memory-kind">{m.kind}</span>
              {m.score != null && (
                <span className="dim">score {m.score.toFixed(3)}</span>
              )}
              <span className="dim">utility {m.utility.toFixed(2)}</span>
              <span className="dim">conf {m.confidence.toFixed(2)}</span>
              <span className="dim">used {m.access_count}x</span>
            </div>
            <p>{m.content}</p>
            {m.tags?.length > 0 && (
              <div className="tags">
                {m.tags.map((t) => (
                  <span key={t} className="tag">
                    {t}
                  </span>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
