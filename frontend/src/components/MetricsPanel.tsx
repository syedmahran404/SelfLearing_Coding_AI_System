import { useEffect, useState } from "react";
import { api } from "../api/client";

/** MetricsPanel — observability dashboard. Polls /metrics every 3s. */
export function MetricsPanel() {
  const [snap, setSnap] = useState<Awaited<ReturnType<typeof api.metrics>> | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const m = await api.metrics();
        if (alive) {
          setSnap(m);
          setError(null);
        }
      } catch (e) {
        if (alive) setError(String((e as Error).message ?? e));
      }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!snap) return <p className="muted">loading metrics…</p>;

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Metrics</h2>
        <span className="dim">uptime {Math.round(snap.uptime_s)}s</span>
      </header>

      <div className="totals">
        <Totals totals={snap.totals} />
      </div>

      <h3 className="sub">By kind</h3>
      <Table data={snap.by_kind} keyHeader="kind" />

      <h3 className="sub">Top names</h3>
      <Table data={topN(snap.by_name, 10)} keyHeader="name" />
    </section>
  );
}

function topN(
  byName: Record<string, Record<string, number>>,
  n: number,
): Record<string, Record<string, number>> {
  const entries = Object.entries(byName).sort((a, b) => {
    const ae = (a[1].ends as number) ?? 0;
    const be = (b[1].ends as number) ?? 0;
    return be - ae;
  });
  return Object.fromEntries(entries.slice(0, n));
}

function Totals({ totals }: { totals: Record<string, number> }) {
  const rows: Array<[string, string]> = [
    ["Spans", String(totals.spans_ended ?? 0)],
    ["Errors", String(totals.errors ?? 0)],
    ["Tokens in", String(totals.tokens_in ?? 0)],
    ["Tokens out", String(totals.tokens_out ?? 0)],
    ["Cost USD", (totals.cost_usd ?? 0).toFixed(6)],
  ];
  return (
    <ul className="totals-row">
      {rows.map(([k, v]) => (
        <li key={k}>
          <span className="dim">{k}</span>
          <strong>{v}</strong>
        </li>
      ))}
    </ul>
  );
}

function Table({
  data,
  keyHeader,
}: {
  data: Record<string, Record<string, number>>;
  keyHeader: string;
}) {
  const cols = ["starts", "ends", "errors", "tokens_in", "tokens_out", "cost_usd", "duration_ms_avg"];
  return (
    <table className="metrics-table">
      <thead>
        <tr>
          <th>{keyHeader}</th>
          {cols.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Object.keys(data).map((name) => (
          <tr key={name}>
            <td className="name">{name}</td>
            {cols.map((c) => (
              <td key={c}>{format(data[name][c])}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function format(v: number | undefined): string {
  if (v === undefined) return "—";
  if (Math.abs(v) < 1 && v !== 0) return v.toFixed(4);
  if (Math.abs(v) < 100) return v.toFixed(2).replace(/\.00$/, "");
  return Math.round(v).toString();
}
