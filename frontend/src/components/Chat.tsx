import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Mode, type Source, type Turn } from "../api";
import Markdown from "./Markdown";
import SourceExcerpt from "./SourceExcerpt";

export interface AssistantMessage {
  role: "assistant";
  id: string;
  question: string;
  status: "searching" | "thinking" | "done";
  sources: Source[];
  answer: string | null;
  error: string | null;
  provider: string | null;
  cited: number[];
  retrieveMs?: number;
  llmS?: number | null;
  startedAt: number;
  mode: Mode;
}
export interface UserMessage {
  role: "user";
  id: string;
  content: string;
}
export type Message = UserMessage | AssistantMessage;

const STARTERS = [
  "What does this project do, and how is it organized?",
  "Where does the program start, and what happens after that?",
  "How is configuration loaded?",
  "How are errors handled?",
];

export const PROVIDER_NAMES: Record<string, string> = {
  claude: "Claude",
  codex: "ChatGPT",
  anthropic: "the Claude API",
  openai: "the OpenAI API",
  ollama: "Ollama",
};

interface Props {
  repoId: string;
  messages: Message[];
  setMessages: (update: (m: Message[]) => Message[]) => void;
  provider: string;
  resolvePath: (text: string) => { path: string; line?: number } | null;
  onOpenFile: (path: string, start?: number, end?: number) => void;
}

let counter = 0;
const newId = () => `${Date.now()}-${counter++}`;

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  return <>{Math.max(0, Math.floor((now - since) / 1000))} s</>;
}

export default function Chat({ repoId, messages, setMessages, provider, resolvePath, onOpenFile }: Props) {
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const threadEnd = useRef<HTMLDivElement>(null);
  const excerptRefs = useRef<Record<number, HTMLElement | null>>({});
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const answers = messages.filter((m): m is AssistantMessage => m.role === "assistant");
  const selected = answers.find((a) => a.id === selectedId) ?? answers[answers.length - 1] ?? null;
  const busy = answers.some((a) => a.status !== "done");

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ block: "end" });
  }, [messages.length]);

  const history = useMemo<Turn[]>(() => {
    const turns: Turn[] = [];
    for (const m of messages) {
      if (m.role === "user") turns.push({ role: "user", content: m.content });
      else if (m.status === "done") turns.push({ role: "assistant", content: m.answer ?? "(no answer)" });
    }
    return turns.slice(-8);
  }, [messages]);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const id = newId();
    const pending: AssistantMessage = {
      role: "assistant", id, question: q, status: "searching", sources: [], answer: null, error: null,
      provider: null, cited: [], startedAt: Date.now(), mode,
    };
    const update = (patch: Partial<AssistantMessage>) =>
      setMessages((ms) => ms.map((m) => (m.id === id && m.role === "assistant" ? { ...m, ...patch } : m)));
    setMessages((ms) => [...ms, { role: "user", id: newId(), content: q }, pending]);
    setSelectedId(id);
    setDraft("");
    try {
      const answer = await api.ask(repoId, { question: q, history, provider, mode }, (s) =>
        update({ sources: s.sources, provider: s.provider, retrieveMs: s.timings.retrieve_ms, status: "thinking", startedAt: Date.now() }),
      );
      update({ status: "done", answer: answer.answer, error: answer.error, cited: answer.cited, llmS: answer.llm_s, provider: answer.provider });
    } catch (e) {
      update({ status: "done", error: (e as Error).message });
    }
    inputRef.current?.focus();
  }

  function focusSource(n: number) {
    const el = excerptRefs.current[n];
    if (!el) return;
    if (selected) setExpanded((x) => ({ ...x, [`${selected.id}:${n}`]: true }));
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setHovered(n);
  }

  const citedSet = new Set(selected?.cited ?? []);
  const hasAnswer = !!selected?.answer;

  return (
    <div className="ask">
      <section className="thread" aria-live="polite">
        {messages.length === 0 && (
          <div className="starters">
            <p className="muted">Ask anything about this codebase. Answers cite the code they come from.</p>
            <div className="starter-list">
              {STARTERS.map((s) => (
                <button key={s} type="button" className="starter" onClick={() => ask(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <p key={m.id} className="question">
              {m.content}
            </p>
          ) : (
            <div
              key={m.id}
              className={`answer${selected?.id === m.id ? " selected" : ""}`}
              onClick={() => setSelectedId(m.id)}
            >
              {m.status === "searching" && <p className="muted">Searching the code</p>}
              {m.status === "thinking" && (
                <p className="muted">
                  Found {m.sources.length} relevant snippets in {m.retrieveMs} ms.{" "}
                  {m.provider && m.provider !== "none" ? (
                    <>
                      {PROVIDER_NAMES[m.provider] ?? m.provider} is reading them (<Elapsed since={m.startedAt} />)
                    </>
                  ) : null}
                </p>
              )}
              {m.status === "done" && m.answer && (
                <Markdown
                  markdown={m.answer}
                  citations={m.sources.length}
                  activeCitation={selected?.id === m.id ? hovered : null}
                  onCiteHover={(n) => {
                    setSelectedId(m.id);
                    setHovered(n);
                  }}
                  onCiteClick={(n) => {
                    setSelectedId(m.id);
                    requestAnimationFrame(() => focusSource(n));
                  }}
                  resolvePath={resolvePath}
                  onOpenPath={(p, line) => onOpenFile(p, line, line)}
                />
              )}
              {m.status === "done" && m.error && (
                <p className={m.provider === "none" ? "notice" : "error-text"}>
                  {m.error}
                  {m.provider === "none" && " Set one up under AI setup at the top right."}
                </p>
              )}
              {m.status === "done" && (
                <p className="answer-meta">
                  {m.answer && m.provider ? `${PROVIDER_NAMES[m.provider] ?? m.provider} answered in ${m.llmS} s. ` : ""}
                  {m.retrieveMs !== undefined ? `Retrieval took ${m.retrieveMs} ms (${m.mode}).` : ""}
                </p>
              )}
            </div>
          ),
        )}
        <div ref={threadEnd} />
      </section>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(draft);
        }}
      >
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          ref={inputRef}
          value={draft}
          rows={2}
          placeholder="Ask how something in this repo works"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask(draft);
            }
          }}
        />
        <div className="composer-row">
          <label className="mode">
            Retrieval
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="hybrid">Hybrid (meaning + keywords)</option>
              <option value="vector">Vector search only</option>
              <option value="keyword">Keyword search only (BM25)</option>
            </select>
          </label>
          <button type="submit" className="primary" disabled={busy || !draft.trim()}>
            {busy ? "Answering" : "Ask"}
          </button>
        </div>
      </form>

      <aside className="sources" aria-label="Sources">
        {!selected && <p className="muted pad">The code behind each answer shows up here.</p>}
        {selected && (
          <>
            <header className="sources-head">
              <h2>Sources</h2>
              <p className="muted">
                {selected.sources.length
                  ? hasAnswer
                    ? `${selected.cited.length} of ${selected.sources.length} cited. Hover a number in the answer to see its lines.`
                    : `${selected.sources.length} closest matches for your question.`
                  : selected.status === "done"
                    ? "Nothing matched."
                    : "Searching"}
              </p>
            </header>
            {selected.sources.map((s) => {
              const cited = !hasAnswer || citedSet.has(s.n);
              const key = `${selected.id}:${s.n}`;
              const collapsed = !cited && !expanded[key];
              return (
                <SourceExcerpt
                  key={key}
                  ref={(el) => {
                    excerptRefs.current[s.n] = el;
                  }}
                  source={s}
                  active={hovered === s.n}
                  cited={cited}
                  collapsed={collapsed}
                  onToggle={cited ? undefined : () => setExpanded((x) => ({ ...x, [key]: !x[key] }))}
                  onOpenFile={onOpenFile}
                  onHover={setHovered}
                />
              );
            })}
          </>
        )}
      </aside>
    </div>
  );
}
