import { useEffect, useState } from "react";
import { api, type Overview as OverviewData, type RepoDetail } from "../api";
import { PROVIDER_NAMES } from "./Chat";
import Markdown from "./Markdown";
import Pipeline from "./Pipeline";

interface Props {
  detail: RepoDetail;
  provider: string;
  providerResolved: string;
  cache: Record<string, OverviewData>;
  setCache: (key: string, value: OverviewData) => void;
  resolvePath: (text: string) => { path: string; line?: number } | null;
  onOpenFile: (path: string, start?: number, end?: number) => void;
}

export default function Overview({ detail, provider, providerResolved, cache, setCache, resolvePath, onOpenFile }: Props) {
  const { meta } = detail;
  const key = `${meta.repo_id}:${meta.indexed_at}:${provider}`;
  const data = cache[key];
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const [, tick] = useState(0);

  async function load(refresh = false) {
    setLoading(true);
    setError(null);
    setStartedAt(Date.now());
    try {
      setCache(key, await api.overview(meta.repo_id, provider, refresh));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!data && !loading) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(() => {
    if (!loading) return;
    const t = setInterval(() => tick((x) => x + 1), 500);
    return () => clearInterval(t);
  }, [loading]);

  const writer = PROVIDER_NAMES[providerResolved] ?? providerResolved;
  const langs = detail.languages.slice(0, 6);

  return (
    <div className="overview">
      <article className="overview-text">
        {loading && (
          <p className="muted">
            {providerResolved === "none" ? "Loading the repo structure" : `${writer} is reading the README, file tree and main files`} (
            {Math.floor((Date.now() - startedAt) / 1000)} s)
          </p>
        )}
        {error && <p className="error-text">{error}</p>}
        {data?.error && <p className="error-text">{data.error}</p>}
        {data && !loading && <Markdown markdown={data.markdown} resolvePath={resolvePath} onOpenPath={(p) => onOpenFile(p)} />}
        {data && !loading && data.provider !== "none" && (
          <div className="overview-foot">
            <span className="muted">Written by {PROVIDER_NAMES[data.provider] ?? data.provider} from the README, file tree and file outlines.</span>
            <button type="button" className="quiet" onClick={() => load(true)}>
              Rewrite overview
            </button>
          </div>
        )}
      </article>

      <aside className="facts" aria-label="Repository facts">
        <section>
          <h2>Size</h2>
          <dl className="numbers">
            <div><dt>Files read</dt><dd>{meta.files.toLocaleString()}</dd></div>
            <div><dt>Lines</dt><dd>{meta.lines.toLocaleString()}</dd></div>
            <div><dt>Chunks</dt><dd>{meta.chunks.toLocaleString()}</dd></div>
          </dl>
        </section>
        {langs.length > 0 && (
          <section>
            <h2>Languages</h2>
            <div className="langbar" aria-hidden="true">
              {langs.map((l, i) => (
                <span key={l.language} style={{ width: `${l.percent}%`, opacity: 1 - i * 0.14 }} />
              ))}
            </div>
            <ul className="langs">
              {langs.map((l) => (
                <li key={l.language}>
                  <span>{l.language}</span>
                  <span className="muted">{l.percent}%</span>
                </li>
              ))}
            </ul>
          </section>
        )}
        {detail.entry_points.length > 0 && (
          <section>
            <h2>Likely entry points</h2>
            <ul className="paths">
              {detail.entry_points.slice(0, 8).map((p) => (
                <li key={p}><button type="button" className="path-link" onClick={() => onOpenFile(p)}><code>{p}</code></button></li>
              ))}
            </ul>
          </section>
        )}
        {detail.key_files.length > 0 && (
          <section>
            <h2>Project files</h2>
            <ul className="paths">
              {detail.key_files.slice(0, 10).map((p) => (
                <li key={p}><button type="button" className="path-link" onClick={() => onOpenFile(p)}><code>{p}</code></button></li>
              ))}
            </ul>
          </section>
        )}
        <section>
          <h2>How it was indexed</h2>
          <Pipeline meta={meta} />
          <p className="muted small">
            Embeddings: {meta.embedding} ({meta.dim} dimensions).
            {meta.reused_vectors > 0 && ` Reused ${meta.reused_vectors} unchanged chunks from the last index.`}
          </p>
        </section>
      </aside>
    </div>
  );
}
