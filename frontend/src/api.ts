// Typed wrappers around the Python API (backend/app/main.py).

export type Stage = "queued" | "cloning" | "scanning" | "chunking" | "embedding" | "saving" | "ready" | "error";

export interface Job {
  repo_id: string;
  name: string;
  source: string;
  stage: Stage;
  progress: number;
  message: string;
  error: string | null;
  failed_at?: string | null;
  running: boolean;
}

export interface Language {
  language: string;
  lines: number;
  percent: number;
}

export interface RepoMeta {
  repo_id: string;
  name: string;
  kind: "github" | "local";
  source: string;
  web_url: string | null;
  commit: string | null;
  indexed_at: string;
  files: number;
  lines: number;
  chunks: number;
  embedding: string;
  dim: number;
  reused_vectors: number;
  languages: Language[];
  skipped: Record<string, number>;
  timings: Record<string, number>;
}

export interface RepoListItem extends Partial<RepoMeta> {
  repo_id: string;
  name: string;
  job: Job | null;
}

export interface RepoDetail {
  meta: RepoMeta;
  job: Job | null;
  languages: Language[];
  tree: string;
  key_files: string[];
  entry_points: string[];
  files: string[];
}

export interface Source {
  n: number;
  path: string;
  start_line: number;
  end_line: number;
  symbol: string;
  context: string;
  kind: string;
  language: string;
  text: string;
  url: string | null;
  score: number;
  vector_rank: number | null;
  keyword_rank: number | null;
  vector_score: number | null;
}

export type Mode = "hybrid" | "vector" | "keyword";

export interface Provider {
  id: string;
  label: string;
  available: boolean;
  setup: string;
}

export interface Health {
  embedding: string;
  llm: { configured: string; default: string; providers: Provider[] };
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export interface SourcesEvent {
  type: "sources";
  query: string;
  mode: Mode;
  sources: Source[];
  provider: string | null;
  timings: { retrieve_ms: number };
}

export interface AnswerEvent {
  type: "answer";
  answer: string | null;
  error: string | null;
  provider: string | null;
  cited: number[];
  llm_s: number | null;
}

export interface Overview {
  markdown: string;
  provider: string;
  cached?: boolean;
  error: string | null;
}

export interface FileText {
  path: string;
  text: string;
  language: string;
  url: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const enc = encodeURIComponent;

export const api = {
  health: () => request<Health>("/api/health"),
  repos: () => request<RepoListItem[]>("/api/repos"),
  index: (source: string) =>
    request<{ repo_id: string; job: Job }>("/api/repos", { method: "POST", body: JSON.stringify({ source }) }),
  detail: (id: string) => request<RepoDetail>(`/api/repos/${enc(id)}`),
  remove: (id: string) => request<{ deleted: string }>(`/api/repos/${enc(id)}`, { method: "DELETE" }),
  overview: (id: string, provider: string, refresh = false) =>
    request<Overview>(`/api/repos/${enc(id)}/overview?provider=${enc(provider)}&refresh=${refresh}`),
  search: (id: string, query: string, mode: Mode, k = 10) =>
    request<{ results: Source[]; retrieve_ms: number }>(`/api/repos/${enc(id)}/search`, {
      method: "POST",
      body: JSON.stringify({ query, mode, k }),
    }),
  file: (id: string, path: string) => request<FileText>(`/api/repos/${enc(id)}/file?path=${enc(path)}`),

  /** Streams the answer: onSources fires as soon as retrieval is done, then the promise resolves with the answer. */
  async ask(
    id: string,
    body: { question: string; history: Turn[]; provider: string; mode: Mode },
    onSources: (e: SourcesEvent) => void,
    signal?: AbortSignal,
  ): Promise<AnswerEvent> {
    const res = await fetch(`/api/repos/${enc(id)}/ask?stream=true`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok || !res.body) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer: AnswerEvent | null = null;
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        const event = JSON.parse(line) as SourcesEvent | AnswerEvent;
        if (event.type === "sources") onSources(event);
        else answer = event;
      }
      if (done) break;
    }
    if (!answer) throw new Error("The server closed the connection before answering.");
    return answer;
  },
};
