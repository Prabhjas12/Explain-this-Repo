import { useEffect, useRef } from "react";
import type { Health } from "../api";

export default function SetupDialog({ health, onClose, onRecheck }: { health: Health; onClose: () => void; onRecheck: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="drawer-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="drawer setup" role="dialog" aria-modal="true" aria-labelledby="setup-title">
        <header className="drawer-head">
          <h2 id="setup-title">AI setup</h2>
          <div className="drawer-actions">
            <button type="button" className="quiet" onClick={onRecheck}>Check again</button>
            <button ref={closeRef} type="button" className="quiet" onClick={onClose}>Close</button>
          </div>
        </header>
        <div className="drawer-body pad">
          <p>
            Search runs entirely on your computer. A language model is only used to write answers from the code that search
            finds. Any one of these works; <em>Auto</em> uses the first one that's ready, from the top.
          </p>
          <ul className="providers">
            {health.llm.providers.map((p) => (
              <li key={p.id} className={p.available ? "ready" : ""}>
                <div className="provider-line">
                  <strong>{p.label}</strong>
                  <span className={p.available ? "status ok" : "status"}>{p.available ? "Ready" : "Not set up"}</span>
                </div>
                {!p.available && p.setup && <code className="setup-cmd">{p.setup}</code>}
              </li>
            ))}
          </ul>
          <p className="muted small">
            Claude and ChatGPT run through their command-line apps with your subscription, so no API key is needed. Their own tools
            for editing files and running commands are switched off; they only see the question and the code snippets. Embeddings:{" "}
            {health.embedding}.
          </p>
        </div>
      </aside>
    </div>
  );
}
