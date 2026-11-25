import type { Job, RepoMeta } from "../api";

const STEPS: { stage: string; name: string; what: string }[] = [
  { stage: "cloning", name: "Clone", what: "Download the repo with git" },
  { stage: "scanning", name: "Scan", what: "Skip binaries, lockfiles and build output" },
  { stage: "chunking", name: "Chunk", what: "Split files at functions and classes" },
  { stage: "embedding", name: "Embed", what: "Turn each chunk into a vector" },
  { stage: "saving", name: "Save", what: "Write the index to disk" },
];
const TIMING_KEYS: Record<string, string> = { cloning: "clone_s", scanning: "scan_s", chunking: "chunk_s", embedding: "embed_s" };

function fmt(s?: number) {
  if (s === undefined) return "";
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`;
}

/** The indexing pipeline as a real sequence of steps: live while indexing, with timings when done. */
export default function Pipeline({ job, meta }: { job?: Job | null; meta?: RepoMeta | null }) {
  const done = !job || job.stage === "ready";
  const failed = job?.stage === "error";
  const stage = failed ? job?.failed_at : job?.stage;
  const current = done ? STEPS.length : Math.max(0, STEPS.findIndex((s) => s.stage === stage));
  return (
    <div className="pipeline">
      <ol className="steps">
        {STEPS.map((step, i) => {
          const state = failed && i === current ? "failed" : i < current ? "done" : i === current && !done ? "now" : "todo";
          const t = meta?.timings?.[TIMING_KEYS[step.stage]];
          return (
            <li key={step.stage} className={`step ${state}`} aria-current={state === "now" ? "step" : undefined}>
              <span className="step-name">{step.name}</span>
              <span className="step-what">{step.what}</span>
              {done && t !== undefined && <span className="step-time">{fmt(t)}</span>}
            </li>
          );
        })}
      </ol>
      {job && !done && (
        <div className="pipeline-status">
          <div className="bar" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <p className={failed ? "error-text" : "muted"}>{failed ? job.error : job.message}</p>
        </div>
      )}
    </div>
  );
}
