import { useState } from "react";
import type { RepoListItem } from "../api";

interface Props {
  repos: RepoListItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onIndex: (source: string) => Promise<void>;
}

export function RepoName({ name }: { name: string }) {
  const i = name.indexOf("/");
  if (i < 0) return <span className="repo-name"><strong>{name}</strong></span>;
  return (
    <span className="repo-name">
      <span className="owner">{name.slice(0, i + 1)}</span>
      <strong>{name.slice(i + 1)}</strong>
    </span>
  );
}

export function AddRepoForm({ onIndex, big = false }: { onIndex: (s: string) => Promise<void>; big?: boolean }) {
  const [source, setSource] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  return (
    <form
      className={big ? "add-form big" : "add-form"}
      onSubmit={async (e) => {
        e.preventDefault();
        if (!source.trim()) return;
        setSending(true);
        setError(null);
        try {
          await onIndex(source.trim());
          setSource("");
        } catch (err) {
          setError((err as Error).message);
        } finally {
          setSending(false);
        }
      }}
    >
      <label htmlFor={big ? "src-big" : "src"} className={big ? "visually-hidden" : "add-label"}>
        Add a repository
      </label>
      <div className="add-row">
        <input
          id={big ? "src-big" : "src"}
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder={big ? "https://github.com/owner/repo, owner/repo, or a folder path" : "owner/repo or URL"}
          spellCheck={false}
          autoComplete="off"
        />
        <button type="submit" className="primary" disabled={sending || !source.trim()}>
          Index
        </button>
      </div>
      {error && <p className="error-text small">{error}</p>}
    </form>
  );
}

export default function Sidebar({ repos, selectedId, onSelect, onIndex }: Props) {
  return (
    <nav className="sidebar" aria-label="Repositories">
      <AddRepoForm onIndex={onIndex} />
      {repos.length > 0 && <h2 className="side-heading">Indexed</h2>}
      <ul className="repo-list">
        {repos.map((r) => {
          const job = r.job;
          const running = job?.running;
          const failed = job?.stage === "error";
          return (
            <li key={r.repo_id}>
              <button
                type="button"
                className={r.repo_id === selectedId ? "repo-item selected" : "repo-item"}
                onClick={() => onSelect(r.repo_id)}
                aria-current={r.repo_id === selectedId ? "page" : undefined}
              >
                <RepoName name={r.name} />
                <span className={failed ? "repo-sub error-text" : "repo-sub"}>
                  {running ? job!.message : failed ? "Indexing failed" : r.files ? `${r.files.toLocaleString()} files, ${r.chunks!.toLocaleString()} chunks` : ""}
                </span>
                {running && (
                  <span className="mini-bar" aria-hidden="true">
                    <span style={{ width: `${Math.round(job!.progress * 100)}%` }} />
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
