import { forwardRef } from "react";
import type { Source } from "../api";
import CodeBlock from "./CodeBlock";

interface Props {
  source: Source;
  active?: boolean;
  cited?: boolean;
  collapsed?: boolean;
  onToggle?: () => void;
  onOpenFile: (path: string, start?: number, end?: number) => void;
  onHover?: (n: number | null) => void;
}

function whyMatched(s: Source): string {
  const parts: string[] = [];
  if (s.vector_rank) parts.push(`#${s.vector_rank} by meaning`);
  if (s.keyword_rank) parts.push(`#${s.keyword_rank} by keywords`);
  return parts.length ? `Found ${parts.join(", ")}` : "";
}

const SourceExcerpt = forwardRef<HTMLElement, Props>(function SourceExcerpt(
  { source: s, active, cited = true, collapsed, onToggle, onOpenFile, onHover },
  ref,
) {
  const label = [s.context, s.symbol].filter(Boolean).join(" › ");
  return (
    <article
      ref={ref}
      className={`excerpt${active ? " active" : ""}${cited ? "" : " uncited"}`}
      onMouseEnter={() => onHover?.(s.n)}
      onMouseLeave={() => onHover?.(null)}
      aria-label={`Source ${s.n}: ${s.path} lines ${s.start_line} to ${s.end_line}`}
    >
      <header className="excerpt-head">
        <span className="excerpt-n">{s.n}</span>
        <div className="excerpt-title">
          <button type="button" className="excerpt-path" onClick={() => onOpenFile(s.path, s.start_line, s.end_line)} title="Open the whole file">
            {s.path}
          </button>
          <span className="excerpt-lines">
            lines {s.start_line}–{s.end_line}
          </span>
          {label && <div className="excerpt-symbol">{label}</div>}
        </div>
        <div className="excerpt-actions">
          {onToggle && (
            <button type="button" className="quiet" onClick={onToggle} aria-expanded={!collapsed}>
              {collapsed ? "Show" : "Hide"}
            </button>
          )}
          {s.url && (
            <a className="quiet" href={s.url} target="_blank" rel="noreferrer">
              GitHub
            </a>
          )}
        </div>
      </header>
      {!collapsed && (
        <>
          <CodeBlock text={s.text} language={s.language} firstLine={s.start_line} mark={active ? [s.start_line, s.end_line] : null} />
          <footer className="excerpt-why">
            {whyMatched(s)}
            {!cited && <span className="excerpt-uncited">Not cited in the answer</span>}
          </footer>
        </>
      )}
    </article>
  );
});

export default SourceExcerpt;
