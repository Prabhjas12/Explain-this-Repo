import { useState } from "react";
import { api, type Mode, type Source } from "../api";
import SourceExcerpt from "./SourceExcerpt";

const MODES: { id: Mode; name: string; help: string }[] = [
  { id: "hybrid", name: "Hybrid", help: "Both lists merged with Reciprocal Rank Fusion. This is what Ask uses." },
  { id: "vector", name: "Vector", help: "Cosine similarity between embeddings. Finds code with the same meaning." },
  { id: "keyword", name: "Keyword", help: "BM25 over code-aware tokens. Finds exact names and rare words." },
];

/** The retrieval half of RAG on its own, so you can see what the model would be shown. */
export default function SearchPanel({ repoId, onOpenFile }: { repoId: string; onOpenFile: (p: string, s?: number, e?: number) => void }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [results, setResults] = useState<Source[] | null>(null);
  const [ms, setMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  async function run(m: Mode = mode) {
    if (!query.trim()) return;
    setError(null);
    try {
      const r = await api.search(repoId, query, m, 10);
      setResults(r.results);
      setMs(r.retrieve_ms);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="search">
      <form
        className="search-form"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <label htmlFor="search-q" className="visually-hidden">Search the code</label>
        <input id="search-q" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="retry failed requests" />
        <button type="submit" className="primary" disabled={!query.trim()}>Search</button>
      </form>
      <div className="segmented" role="radiogroup" aria-label="Retrieval method">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            role="radio"
            aria-checked={mode === m.id}
            className={mode === m.id ? "on" : ""}
            onClick={() => {
              setMode(m.id);
              if (results) run(m.id);
            }}
          >
            {m.name}
          </button>
        ))}
      </div>
      <p className="muted small">{MODES.find((m) => m.id === mode)!.help} Switch methods to compare what each one finds.</p>
      {error && <p className="error-text">{error}</p>}
      {results && <p className="muted small">{results.length} results in {ms} ms.</p>}
      {results?.length === 0 && <p className="notice">No matches. Try different words, or Vector search for a description in plain English.</p>}
      <div className="search-results">
        {results?.map((s) => (
          <SourceExcerpt key={`${mode}:${s.n}`} source={s} active={hovered === s.n} onHover={setHovered} onOpenFile={onOpenFile} />
        ))}
      </div>
    </div>
  );
}
