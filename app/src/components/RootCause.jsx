import { GitBranch, Wrench, FileText, CheckCircle2 } from "lucide-react";

export default function RootCause({ inspection, onOpenReport }) {
  if (!inspection) {
    return (
      <div className="panel rounded-2xl p-6 rise-in">
        <div className="flex items-center gap-2.5 mb-2">
          <GitBranch size={17} className="text-[var(--accent-cyan)]" />
          <h2 className="font-display font-semibold text-[15px]">Root Cause &amp; Recommendation</h2>
        </div>
        <p className="text-sm text-[var(--text-dim)]">Nothing to analyze yet.</p>
      </div>
    );
  }

  const { report, risk, status } = inspection;

  return (
    <div className="panel rounded-2xl p-6 flex flex-col rise-in">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <GitBranch size={17} className="text-[var(--accent-cyan)]" />
          <h2 className="font-display font-semibold text-[15px]">Root Cause &amp; Recommendation</h2>
        </div>
        {status !== "pending_review" && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--accent-green)]">
            <CheckCircle2 size={13} />
            {status === "approved" ? "Approved" : "Revised"}
          </span>
        )}
      </div>

      <div className="grid grid-cols-[1.4fr_1fr] gap-6 mb-5">
        <div>
          <div className="text-xs font-medium text-[var(--accent-violet)] mb-1.5">
            Probable Root Cause
          </div>
          <p className="text-sm text-[var(--text-muted)] leading-relaxed mb-4">
            {report.root_cause}
          </p>

          <div className="text-xs font-medium text-[var(--accent-violet)] mb-1.5">
            Recommended Action
          </div>
          <ol className="text-sm text-[var(--text-muted)] leading-relaxed list-decimal list-inside space-y-1">
            {report.recommended_next_steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </div>

        <div className="flex flex-col items-center justify-center gap-3 border-l border-[var(--border-soft)] pl-6">
          <div className="w-14 h-14 rounded-full bg-[var(--accent-violet-dim)] border border-[#8b6cf84d] flex items-center justify-center">
            <Wrench size={22} className="text-[var(--accent-violet)]" />
          </div>
          <div className="text-center">
            <div className="text-[11px] text-[var(--text-dim)]">Maintenance Priority</div>
            <div className="text-sm font-semibold text-[var(--accent-red)]">{risk.priority}</div>
          </div>
          <div className="text-center">
            <div className="text-[11px] text-[var(--text-dim)]">Suggested Downtime</div>
            <div className="text-sm font-semibold">{risk.downtime}</div>
          </div>
        </div>
      </div>

      <button
        onClick={onOpenReport}
        className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-gradient-to-br from-[#8b6cf8] to-[#6a4ef0] text-sm font-medium hover:opacity-90 transition-opacity panel-glow"
      >
        <FileText size={15} />
        View Detailed Report
      </button>
    </div>
  );
}
