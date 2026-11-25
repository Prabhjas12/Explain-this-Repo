import { Children, isValidElement, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CodeBlock from "./CodeBlock";

interface Props {
  markdown: string;
  citations?: number; // how many numbered sources exist; [n] up to this becomes a chip
  activeCitation?: number | null;
  onCiteHover?: (n: number | null) => void;
  onCiteClick?: (n: number) => void;
  resolvePath?: (text: string) => { path: string; line?: number } | null;
  onOpenPath?: (path: string, line?: number) => void;
}

/** Turn [3] into a link we can render as a chip. Code spans and fences are left alone. */
function linkCitations(md: string, max: number): string {
  if (!max) return md;
  return md
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(/\[(\d{1,2})\](?!\()/g, (m, d) => (Number(d) >= 1 && Number(d) <= max ? `[${d}](#cite-${d})` : m)),
    )
    .join("");
}

function textOf(children: ReactNode): string {
  return Children.toArray(children)
    .map((c) => (typeof c === "string" || typeof c === "number" ? String(c) : isValidElement(c) ? textOf((c.props as { children?: ReactNode }).children) : ""))
    .join("");
}

export default function Markdown({ markdown, citations = 0, activeCitation, onCiteHover, onCiteClick, resolvePath, onOpenPath }: Props) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            const m = href?.match(/^#cite-(\d+)$/);
            if (m) {
              const n = Number(m[1]);
              return (
                <button
                  type="button"
                  className={activeCitation === n ? "cite active" : "cite"}
                  onMouseEnter={() => onCiteHover?.(n)}
                  onMouseLeave={() => onCiteHover?.(null)}
                  onFocus={() => onCiteHover?.(n)}
                  onBlur={() => onCiteHover?.(null)}
                  onClick={() => onCiteClick?.(n)}
                  aria-label={`Source ${n}`}
                >
                  {n}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
          pre({ children }) {
            return <>{children}</>;
          },
          code({ className, children }) {
            const text = textOf(children).replace(/\n$/, "");
            const lang = /language-(\w+)/.exec(className || "")?.[1];
            if (lang || text.includes("\n")) {
              return (
                <div className="prose-code">
                  <CodeBlock text={text} language={lang || "plaintext"} />
                </div>
              );
            }
            const hit = resolvePath?.(text);
            if (hit && onOpenPath) {
              return (
                <button type="button" className="path-link" onClick={() => onOpenPath(hit.path, hit.line)} title={`Open ${hit.path}`}>
                  <code>{text}</code>
                </button>
              );
            }
            return <code>{children}</code>;
          },
        }}
      >
        {linkCitations(markdown, citations)}
      </ReactMarkdown>
    </div>
  );
}
