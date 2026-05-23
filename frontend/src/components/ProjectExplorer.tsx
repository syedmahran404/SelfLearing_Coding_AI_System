import { FormEvent, useEffect, useState } from "react";
import { ProjectRow, api } from "../api/client";

interface Props {
  selectedId?: string;
  onSelect?: (id: string | undefined) => void;
}

/** ProjectExplorer — list / create / select projects. */
export function ProjectExplorer({ selectedId, onSelect }: Props) {
  const [rows, setRows] = useState<ProjectRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // create form
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [repoRoot, setRepoRoot] = useState("");

  const refresh = async () => {
    setLoading(true);
    try {
      const data = await api.listProjects();
      setRows(data);
      setError(null);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !slug.trim()) return;
    try {
      await api.createProject({
        name: name.trim(),
        slug: slug.trim(),
        repo_root: repoRoot.trim() || undefined,
      });
      setName("");
      setSlug("");
      setRepoRoot("");
      setShowCreate(false);
      refresh();
    } catch (err) {
      setError(String((err as Error).message ?? err));
    }
  };

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Projects</h2>
        <button
          className="ghost small"
          onClick={() => setShowCreate((v) => !v)}
          type="button"
        >
          {showCreate ? "Cancel" : "New project"}
        </button>
      </header>

      {error && <p className="error">{error}</p>}

      {showCreate && (
        <form onSubmit={create} className="project-create">
          <input
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
          <input
            placeholder="slug-like-this"
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
            pattern="[a-z0-9][a-z0-9-_]*"
            required
          />
          <input
            placeholder="/abs/path/to/repo (optional)"
            value={repoRoot}
            onChange={(e) => setRepoRoot(e.target.value)}
          />
          <button type="submit">Create</button>
        </form>
      )}

      {loading ? (
        <p className="muted">loading…</p>
      ) : rows.length === 0 ? (
        <p className="muted">No projects yet.</p>
      ) : (
        <ul className="project-list">
          <li
            className={`project-item ${selectedId == null ? "selected" : ""}`}
            onClick={() => onSelect?.(undefined)}
          >
            <strong>(none)</strong>
            <span className="dim">no project context</span>
          </li>
          {rows.map((p) => (
            <li
              key={p.id}
              className={`project-item ${selectedId === p.id ? "selected" : ""}`}
              onClick={() => onSelect?.(p.id)}
            >
              <strong>{p.name}</strong>
              <span className="dim">{p.slug}</span>
              {p.repo_root && <span className="dim">{p.repo_root}</span>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
