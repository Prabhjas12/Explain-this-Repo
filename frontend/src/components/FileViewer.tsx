import { useEffect, useRef, useState } from "react";
import { api, type FileText } from "../api";
import CodeBlock from "./CodeBlock";

interface Props {
  repoId: string;
  path: string;
  start?: number;
  end?: number;
  onClose: () => void;
}

export default function FileViewer({ repoId, path, start, end, onClose }: Props) {
  const [file, setFile] = useState<FileText | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const markEl = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setFile(null);
    setError(null);
    api.file(repoId, path).then(setFile, (e: Error) => setError(e.message));
  }, [repoId, path]);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    if (file && markEl.current) markEl.current.scrollIntoView({ block: "center" });
  }, [file]);

  const mark: [number, number] | null = start ? [start, end ?? start] : null;
  const url = file?.url && start ? `${file.url}#L${start}${end && end !== start ? `-L${end}` : ""}` : file?.url;

  return (
    <div className="drawer-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={path}>
        <header className="drawer-head">
          <div>
            <div className="drawer-path">{path}</div>
            {mark && (
              <div className="drawer-sub">
                Highlighted: lines {mark[0]}–{mark[1]}
              </div>
            )}
          </div>
          <div className="drawer-actions">
            {url && (
              <a className="quiet" href={url} target="_blank" rel="noreferrer">
                Open on GitHub
              </a>
            )}
            <button ref={closeRef} type="button" className="quiet" onClick={onClose}>
              Close
            </button>
          </div>
        </header>
        <div className="drawer-body">
          {error && <p className="error-text">{error}</p>}
          {!file && !error && <p className="muted pad">Loading {path}</p>}
          {file && <CodeBlock text={file.text} language={file.language} mark={mark} markRef={(el) => (markEl.current = el)} />}
        </div>
      </aside>
    </div>
  );
}
