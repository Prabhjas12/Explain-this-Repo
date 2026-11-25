import { useState } from "react";
import CodeBlock from "./CodeBlock";
import Markdown from "./Markdown";
import { AddRepoForm } from "./Sidebar";

// A real excerpt from this project (backend/app/search.py), so the demo is honest.
const DEMO_CODE = `    else:
        fused = defaultdict(float)
        for i, r in vec_rank.items():
            fused[i] += 1.0 / (RRF_K + r)
        for i, r in kw_rank.items():
            fused[i] += 1.0 / (RRF_K + r)
        if doc_weight != 1.0 or test_weight != 1.0:
            for i in fused:
                fused[i] *= prior(chunks[i], doc_weight, test_weight)
        order = sorted(fused, key=lambda i: (-fused[i], min(vec_rank.get(i, 999), kw_rank.get(i, 999))))`;
const DEMO_FIRST = 196;
const DEMO_MARKS: Record<number, [number, number]> = { 1: [197, 201], 2: [202, 204], 3: [205, 205] };
const DEMO_ANSWER =
  "It uses Reciprocal Rank Fusion. Every chunk earns `1 / (60 + rank)` from each list it shows up in, and the two are added together [1]. Docs and tests then get scaled down slightly, so source code wins close calls [2]. Finally the chunks are sorted by that score, with the best single rank breaking ties [3].";

const EXAMPLES = ["pallets/flask", "psf/requests", "expressjs/express", "Prabhjas12/re-audit"];

export default function Welcome({ onIndex }: { onIndex: (s: string) => Promise<void> }) {
  const [cite, setCite] = useState<number | null>(1);
  return (
    <div className="welcome">
      <div className="welcome-intro">
        <h1>Point it at a repository. Ask how it works.</h1>
        <p className="lede">
          Explain This Repo reads a codebase, splits it at its functions and classes, and indexes it on your computer. Every
          answer cites the exact lines it came from, so you can check it.
        </p>
        <AddRepoForm onIndex={onIndex} big />
        <div className="examples">
          <span className="muted">Try</span>
          {EXAMPLES.map((e) => (
            <button key={e} type="button" className="example" onClick={() => onIndex(e)}>
              {e}
            </button>
          ))}
        </div>
      </div>

      <figure className="demo" aria-label="Example answer with citations">
        <div className="demo-answer">
          <p className="question">How does hybrid search combine the two result lists?</p>
          <Markdown markdown={DEMO_ANSWER} citations={3} activeCitation={cite} onCiteHover={(n) => n && setCite(n)} onCiteClick={setCite} />
          <figcaption className="muted small">Hover the numbers. This answer is about this app's own code.</figcaption>
        </div>
        <div className="demo-code">
          <div className="excerpt-head">
            <span className="excerpt-n">{cite ?? 1}</span>
            <div className="excerpt-title">
              <span className="excerpt-path">backend/app/search.py</span>
              <span className="excerpt-lines">lines {DEMO_FIRST}–{DEMO_FIRST + 9}</span>
            </div>
          </div>
          <CodeBlock text={DEMO_CODE} language="python" firstLine={DEMO_FIRST} mark={cite ? DEMO_MARKS[cite] : null} />
        </div>
      </figure>
    </div>
  );
}
