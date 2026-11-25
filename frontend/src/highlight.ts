import hljs from "highlight.js/lib/common";

const ALIASES: Record<string, string> = {
  notebook: "python",
  html: "xml",
  vue: "xml",
  svelte: "xml",
  toml: "ini",
  dockerfile: "bash",
  terraform: "ini",
  text: "plaintext",
  rst: "plaintext",
};

function languageFor(lang: string): string {
  const name = ALIASES[lang] ?? lang;
  return hljs.getLanguage(name) ? name : "plaintext";
}

/**
 * Highlight code and return one HTML string per line.
 * Spans that cross a line break (block comments, multi-line strings) are closed at the
 * end of each line and reopened on the next, so every line is valid HTML on its own.
 */
export function highlightLines(code: string, lang: string): string[] {
  let html: string;
  try {
    html = hljs.highlight(code, { language: languageFor(lang), ignoreIllegals: true }).value;
  } catch {
    html = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  const lines: string[] = [];
  const open: string[] = [];
  let current = "";
  for (const token of html.split(/(<span[^>]*>|<\/span>|\n)/)) {
    if (!token) continue;
    if (token === "\n") {
      lines.push(current + "</span>".repeat(open.length));
      current = open.join("");
    } else if (token.startsWith("<span")) {
      open.push(token);
      current += token;
    } else if (token === "</span>") {
      open.pop();
      current += token;
    } else {
      current += token;
    }
  }
  lines.push(current + "</span>".repeat(open.length));
  return lines;
}
