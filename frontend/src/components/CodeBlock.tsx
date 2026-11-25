import { useMemo } from "react";
import { highlightLines } from "../highlight";

interface Props {
  text: string;
  language: string;
  firstLine?: number;
  /** absolute line numbers to mark with the highlighter */
  mark?: [number, number] | null;
  /** receives the first marked line, so a parent can scroll to it */
  markRef?: (el: HTMLDivElement | null) => void;
}

export default function CodeBlock({ text, language, firstLine = 1, mark, markRef }: Props) {
  const lines = useMemo(() => highlightLines(text, language), [text, language]);
  const firstMarked = mark ? Math.max(mark[0], firstLine) : -1;
  return (
    <div className="code">
      {lines.map((html, i) => {
        const n = firstLine + i;
        const marked = !!mark && n >= mark[0] && n <= mark[1];
        return (
          <div className={marked ? "code-line marked" : "code-line"} key={n} ref={n === firstMarked ? markRef : undefined}>
            <span className="gutter" aria-hidden="true">
              {n}
            </span>
            <span className="src" dangerouslySetInnerHTML={{ __html: html || " " }} />
          </div>
        );
      })}
    </div>
  );
}
