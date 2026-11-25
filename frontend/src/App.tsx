import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Health, type Overview as OverviewData, type RepoDetail, type RepoListItem } from "./api";
import Chat, { PROVIDER_NAMES, type Message } from "./components/Chat";
import FileViewer from "./components/FileViewer";
import Overview from "./components/Overview";
import Pipeline from "./components/Pipeline";
import SearchPanel from "./components/SearchPanel";
import SetupDialog from "./components/SetupDialog";
import Sidebar, { RepoName } from "./components/Sidebar";
import Welcome from "./components/Welcome";

type Tab = "overview" | "ask" | "search";

const store = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string | null) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      /* storage unavailable */
    }
  },
};

function makeResolver(files: string[]) {
  const all = new Set(files);
  const byName = new Map<string, string[]>();
  for (const f of files) {
    const base = f.slice(f.lastIndexOf("/") + 1);
    byName.set(base, [...(byName.get(base) ?? []), f]);
  }
  return (raw: string): { path: string; line?: number } | null => {
    const m = raw.trim().match(/^\.?\/?([\w@.\-/]+?\.\w+|[\w.\-/]+\/)(?::(\d+)(?:-\d+)?|#L(\d+))?$/);
    if (!m) return null;
    const p = m[1].replace(/\/$/, "");
    const line = Number(m[2] || m[3]) || undefined;
    if (all.has(p)) return { path: p, line };
    const suffix = files.filter((f) => f.endsWith("/" + p));
    if (suffix.length === 1) return { path: suffix[0], line };
    const named = byName.get(p);
    if (named?.length === 1) return { path: named[0], line };
    return null;
  };
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [repos, setRepos] = useState<RepoListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(() => store.get("etr.repo"));
  const [provider, setProvider] = useState<string>(() => store.get("etr.provider") || "auto");
  const [details, setDetails] = useState<Record<string, RepoDetail>>({});
  const [chats, setChats] = useState<Record<string, Message[]>>({});
  const [overviews, setOverviews] = useState<Record<string, OverviewData>>({});
  const [tabs, setTabs] = useState<Record<string, Tab>>({});
  const [file, setFile] = useState<{ path: string; start?: number; end?: number } | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refreshRepos = useCallback(async () => {
    try {
      setRepos(await api.repos());
      setLoadError(null);
    } catch (e) {
      setLoadError(`Can't reach the Explain This Repo server (${(e as Error).message}). Is ./run.sh still running?`);
    }
  }, []);
  const refreshHealth = useCallback(() => api.health().then(setHealth, () => undefined), []);

  useEffect(() => {
    refreshRepos();
    refreshHealth();
  }, [refreshRepos, refreshHealth]);

  const anyRunning = repos.some((r) => r.job?.running);
  useEffect(() => {
    if (!anyRunning) return;
    const t = setInterval(refreshRepos, 700);
    return () => clearInterval(t);
  }, [anyRunning, refreshRepos]);

  const selected = repos.find((r) => r.repo_id === selectedId) ?? null;
  useEffect(() => store.set("etr.repo", selectedId), [selectedId]);
  useEffect(() => store.set("etr.provider", provider), [provider]);

  const indexedAt = selected?.indexed_at;
  useEffect(() => {
    if (!selectedId || !indexedAt) return;
    api.detail(selectedId).then((d) => setDetails((x) => ({ ...x, [selectedId]: d })), () => undefined);
  }, [selectedId, indexedAt]);

  const detail = selectedId ? details[selectedId] : undefined;
  const fresh = detail && detail.meta.indexed_at === indexedAt ? detail : undefined;
  const resolvePath = useMemo(() => makeResolver(fresh?.files ?? []), [fresh]);

  async function index(source: string) {
    const r = await api.index(source);
    setSelectedId(r.repo_id);
    setConfirmRemove(false);
    await refreshRepos();
  }

  async function remove(id: string) {
    await api.remove(id);
    setSelectedId(null);
    setConfirmRemove(false);
    setChats((c) => ({ ...c, [id]: [] }));
    await refreshRepos();
  }

  const resolvedProvider =
    provider === "auto" ? health?.llm.default ?? "none" : provider;
  const tab: Tab = (selectedId && tabs[selectedId]) || "overview";
  const job = selected?.job;
  const openFile = (path: string, start?: number, end?: number) => setFile({ path, start, end });

  return (
    <div className="app">
      <header className="topbar">
        <button type="button" className="brand" onClick={() => setSelectedId(null)}>
          Explain This Repo
        </button>
        <div className="topbar-right">
          {health && (
            <label className="provider-select">
              <span>Answers by</span>
              <select value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="auto">Auto ({resolvedProviderLabel(health, health.llm.default)})</option>
                {health.llm.providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                    {p.available ? "" : " (not set up)"}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button type="button" className="quiet" onClick={() => setShowSetup(true)} disabled={!health}>
            AI setup
          </button>
        </div>
      </header>

      <Sidebar repos={repos} selectedId={selectedId} onSelect={(id) => { setSelectedId(id); setConfirmRemove(false); }} onIndex={index} />

      <main className="main">
        {loadError && <p className="error-text pad">{loadError}</p>}
        {!selected && <Welcome onIndex={index} />}

        {selected && (
          <>
            <div className="repo-head">
              <div>
                <h1>
                  <RepoName name={selected.name} />
                </h1>
                <p className="repo-facts muted">
                  {selected.commit && (
                    <>
                      Commit <code>{selected.commit.slice(0, 7)}</code>.{" "}
                    </>
                  )}
                  {selected.indexed_at && <>Indexed {new Date(selected.indexed_at).toLocaleString()}.</>}
                </p>
              </div>
              <div className="repo-actions">
                {selected.web_url && (
                  <a className="quiet" href={selected.web_url} target="_blank" rel="noreferrer">
                    GitHub
                  </a>
                )}
                {selected.indexed_at && !job?.running && (
                  <button type="button" className="quiet" onClick={() => index(selected.source || selected.name)}>
                    Re-index
                  </button>
                )}
                {selected.indexed_at && !job?.running &&
                  (confirmRemove ? (
                    <span className="confirm">
                      Remove this index?
                      <button type="button" className="quiet danger" onClick={() => remove(selected.repo_id)}>Remove</button>
                      <button type="button" className="quiet" onClick={() => setConfirmRemove(false)}>Keep</button>
                    </span>
                  ) : (
                    <button type="button" className="quiet" onClick={() => setConfirmRemove(true)}>
                      Remove
                    </button>
                  ))}
              </div>
            </div>

            {(job?.running || job?.stage === "error") && (
              <div className="indexing">
                <h2>{job.stage === "error" ? "Indexing stopped" : selected.indexed_at ? "Re-indexing" : "Indexing"}</h2>
                <Pipeline job={job} />
                {job.stage === "error" && (
                  <button type="button" className="primary" onClick={() => index(job.source || selected.source || selected.name)}>
                    Try again
                  </button>
                )}
              </div>
            )}

            {selected.indexed_at && fresh && !job?.running && (
              <>
                <div className="tabs" role="tablist">
                  {(["overview", "ask", "search"] as Tab[]).map((t) => (
                    <button
                      key={t}
                      type="button"
                      role="tab"
                      aria-selected={tab === t}
                      className={tab === t ? "tab on" : "tab"}
                      onClick={() => setTabs((x) => ({ ...x, [selected.repo_id]: t }))}
                    >
                      {t === "overview" ? "Overview" : t === "ask" ? "Ask" : "Search"}
                    </button>
                  ))}
                </div>
                <div className="tab-body" role="tabpanel">
                  {tab === "overview" && (
                    <Overview
                      detail={fresh}
                      provider={provider}
                      providerResolved={resolvedProvider}
                      cache={overviews}
                      setCache={(k, v) => setOverviews((x) => ({ ...x, [k]: v }))}
                      resolvePath={resolvePath}
                      onOpenFile={openFile}
                    />
                  )}
                  {tab === "ask" && (
                    <Chat
                      key={selected.repo_id}
                      repoId={selected.repo_id}
                      messages={chats[selected.repo_id] ?? []}
                      setMessages={(u) => setChats((c) => ({ ...c, [selected.repo_id]: u(c[selected.repo_id] ?? []) }))}
                      provider={provider}
                      resolvePath={resolvePath}
                      onOpenFile={openFile}
                    />
                  )}
                  {tab === "search" && <SearchPanel repoId={selected.repo_id} onOpenFile={openFile} />}
                </div>
              </>
            )}
          </>
        )}
      </main>

      {file && selected && <FileViewer repoId={selected.repo_id} {...file} onClose={() => setFile(null)} />}
      {showSetup && health && <SetupDialog health={health} onClose={() => setShowSetup(false)} onRecheck={refreshHealth} />}
    </div>
  );
}

function resolvedProviderLabel(health: Health, id: string) {
  if (id === "none") return "retrieval only";
  return PROVIDER_NAMES[id] ?? health.llm.providers.find((p) => p.id === id)?.label ?? id;
}
