import { useState } from "react";
import { X, CheckCircle2, PencilLine, Download, Printer, ShieldCheck, Loader2 } from "lucide-react";
import { approveInspection, reviseInspection } from "../api/pipeline";

export default function ReportModal({ inspection, onClose, onUpdate }) {
  const [editing, setEditing] = useState(false);
  const [rootCause, setRootCause] = useState(inspection.report.root_cause);
  const [steps, setSteps] = useState(inspection.report.recommended_next_steps.join("\n"));
  const [busy, setBusy] = useState(false);
  const [reviewer, setReviewer] = useState("");

  const handleApprove = async () => {
    if (!reviewer.trim()) return;
    setBusy(true);
    const res = await approveInspection(inspection.inspection_id, reviewer.trim());
    setBusy(false);
    onUpdate({ ...inspection, status: "approved", reviewer: res.reviewer, reviewed_at: res.reviewed_at });
    onClose();
  };

  const handleSaveRevision = async () => {
    setBusy(true);
    const revised = {
      root_cause: rootCause,
      recommended_next_steps: steps.split("\n").filter(Boolean),
    };
    await reviseInspection(inspection.inspection_id, revised);
    setBusy(false);
    onUpdate({
      ...inspection,
      status: "revised",
      report: { ...inspection.report, ...revised },
    });
    setEditing(false);
  };

  const handleExportJSON = () => {
    const blob = new Blob([JSON.stringify(inspection, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${inspection.inspection_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-6">
      <div className="w-full max-w-2xl max-h-[88vh] overflow-y-auto panel rounded-2xl panel-glow">
        <div id="print-report" className="p-7">
          <div className="flex items-start justify-between mb-6">
            <div>
              <div className="text-[11px] font-mono text-[var(--text-dim)] mb-1">
                {inspection.inspection_id}
              </div>
              <h2 className="font-display text-xl font-bold">Inspection Report</h2>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-lg hover:bg-white/[0.05] flex items-center justify-center text-[var(--text-muted)] print:hidden"
            >
              <X size={16} />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
            <div>
              <div className="text-[11px] text-[var(--text-dim)]">Component</div>
              <div className="font-medium">{inspection.component}</div>
            </div>
            <div>
              <div className="text-[11px] text-[var(--text-dim)]">Line</div>
              <div className="font-medium">{inspection.line}</div>
            </div>
            <div>
              <div className="text-[11px] text-[var(--text-dim)]">Predicted Defect</div>
              <div className="font-medium">{inspection.vision_output.label}</div>
            </div>
            <div>
              <div className="text-[11px] text-[var(--text-dim)]">Confidence</div>
              <div className="font-medium">
                {(inspection.vision_output.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="mb-5">
            <div className="text-xs font-medium text-[var(--accent-violet)] mb-1.5">Impression</div>
            <p className="text-sm text-[var(--text-muted)] leading-relaxed">
              {inspection.report.impression}
            </p>
          </div>

          <div className="mb-5">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-medium text-[var(--accent-violet)]">
                Root Cause &amp; Recommended Action
              </span>
              {!editing && inspection.status === "pending_review" && (
                <button
                  onClick={() => setEditing(true)}
                  className="text-xs flex items-center gap-1 text-[var(--text-muted)] hover:text-white print:hidden"
                >
                  <PencilLine size={12} /> Revise
                </button>
              )}
            </div>

            {editing ? (
              <div className="space-y-2">
                <textarea
                  value={rootCause}
                  onChange={(e) => setRootCause(e.target.value)}
                  rows={3}
                  className="w-full text-sm bg-[#0a0e1c] border border-[var(--border-soft)] rounded-lg p-3 text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-violet)]"
                />
                <textarea
                  value={steps}
                  onChange={(e) => setSteps(e.target.value)}
                  rows={3}
                  className="w-full text-sm bg-[#0a0e1c] border border-[var(--border-soft)] rounded-lg p-3 text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-violet)]"
                  placeholder="One action per line"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveRevision}
                    disabled={busy}
                    className="px-3 py-1.5 rounded-lg bg-[var(--accent-violet-dim)] border border-[#8b6cf84d] text-xs font-medium flex items-center gap-1.5"
                  >
                    {busy && <Loader2 size={12} className="animate-spin" />}
                    Save revision
                  </button>
                  <button
                    onClick={() => setEditing(false)}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--text-muted)]"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <>
                <p className="text-sm text-[var(--text-muted)] leading-relaxed mb-2">
                  {inspection.report.root_cause}
                </p>
                <ol className="text-sm text-[var(--text-muted)] leading-relaxed list-decimal list-inside space-y-1">
                  {inspection.report.recommended_next_steps.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ol>
              </>
            )}
          </div>

          <div className="mb-6">
            <div className="text-xs font-medium text-[var(--accent-violet)] mb-1.5">
              Supporting Evidence
            </div>
            <ul className="text-sm text-[var(--text-muted)] leading-relaxed list-disc list-inside space-y-1">
              {inspection.report.supporting_evidence.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          </div>

          <div className="flex items-center gap-2 mb-6 p-3 rounded-xl bg-[#34d39914] border border-[#34d39933]">
            <ShieldCheck size={16} className="text-[var(--accent-green)] shrink-0" />
            <span className="text-xs text-[var(--text-muted)]">
              Verification agent confidence:{" "}
              <span className="text-[var(--accent-green)] font-medium">
                {inspection.verification.confidence_score}/100
              </span>{" "}
              — no unsupported claims flagged.
            </span>
          </div>

          {/* Human-in-the-loop gate */}
          {inspection.status === "pending_review" ? (
            <div className="border-t border-[var(--border-soft)] pt-5 print:hidden">
              <div className="text-xs font-medium text-white mb-2">
                Reviewer sign-off required before this report is finalized
              </div>
              <input
                value={reviewer}
                onChange={(e) => setReviewer(e.target.value)}
                placeholder="Your name"
                className="w-full text-sm bg-[#0a0e1c] border border-[var(--border-soft)] rounded-lg p-2.5 mb-3 focus:outline-none focus:border-[var(--accent-violet)]"
              />
              <button
                onClick={handleApprove}
                disabled={busy || !reviewer.trim()}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-gradient-to-br from-[#8b6cf8] to-[#6a4ef0] text-sm font-medium disabled:opacity-40 panel-glow"
              >
                {busy ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle2 size={15} />}
                Approve Report
              </button>
            </div>
          ) : (
            <div className="border-t border-[var(--border-soft)] pt-4 flex items-center gap-2 text-sm print:hidden">
              <CheckCircle2 size={15} className="text-[var(--accent-green)]" />
              <span>
                {inspection.status === "approved" ? "Approved" : "Revised"}
                {inspection.reviewer ? ` by ${inspection.reviewer}` : ""}
              </span>
            </div>
          )}
        </div>

        <div className="flex gap-2 px-7 pb-6 print:hidden">
          <button
            onClick={handleExportJSON}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg border border-[var(--border-soft)] text-xs font-medium text-[var(--text-muted)] hover:text-white transition-colors"
          >
            <Download size={13} /> Export JSON
          </button>
          <button
            onClick={() => window.print()}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg border border-[var(--border-soft)] text-xs font-medium text-[var(--text-muted)] hover:text-white transition-colors"
          >
            <Printer size={13} /> Print / Save PDF
          </button>
        </div>
      </div>
    </div>
  );
}
