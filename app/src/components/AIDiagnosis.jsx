import { Sparkles } from "lucide-react";
import ConfidenceGauge from "./ConfidenceGauge";

const SEVERITY_STYLE = {
  High: "bg-[#f0526a1f] text-[var(--accent-red)] border-[#f0526a40]",
  Medium: "bg-[#f5a5241f] text-[var(--accent-amber)] border-[#f5a52440]",
  Low: "bg-[#34d3991f] text-[var(--accent-green)] border-[#34d39940]",
};

export default function AIDiagnosis({ inspection }) {
  if (!inspection) {
    return (
      <div className="panel rounded-2xl p-6 rise-in">
        <div className="flex items-center gap-2.5 mb-2">
          <Sparkles size={17} className="text-[var(--accent-violet)]" />
          <h2 className="font-display font-semibold text-[15px]">AI Diagnosis</h2>
        </div>
        <p className="text-sm text-[var(--text-dim)]">Run an inspection to see a diagnosis.</p>
      </div>
    );
  }

  const { vision_output, risk } = inspection;
  const areaPx = Math.round(
    vision_output.bbox[2] * vision_output.bbox[3] * 1_000_000 * 0.001
  );

  return (
    <div className="panel rounded-2xl p-6 rise-in">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <Sparkles size={17} className="text-[var(--accent-violet)]" />
          <h2 className="font-display font-semibold text-[15px]">AI Diagnosis</h2>
        </div>
        <span
          className={`text-xs font-medium px-2.5 py-1 rounded-full border ${SEVERITY_STYLE[risk.severity]}`}
        >
          {risk.severity} Severity
        </span>
      </div>

      <div className="flex items-center gap-5 mb-5">
        <div className="flex-1">
          <div className="text-[11px] text-[var(--text-dim)] mb-1">Predicted Defect</div>
          <div className="font-display text-xl font-bold mb-3">{vision_output.label}</div>
          <div className="text-[11px] text-[var(--text-dim)] mb-0.5">Confidence Score</div>
          <div className="text-xs font-mono text-[var(--text-muted)]">
            X: {vision_output.bbox[0]}, Y: {vision_output.bbox[1]}, W: {vision_output.bbox[2]}, H:{" "}
            {vision_output.bbox[3]}
          </div>
        </div>
        <ConfidenceGauge value={vision_output.confidence} />
      </div>

      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-[var(--border-soft)]">
        <div>
          <div className="text-[11px] text-[var(--text-dim)] mb-0.5">Defect Area</div>
          <div className="text-sm font-medium">{areaPx.toLocaleString()} px²</div>
        </div>
        <div>
          <div className="text-[11px] text-[var(--text-dim)] mb-0.5">Failure Probability</div>
          <div className="text-sm font-medium">{risk.failure_probability}%</div>
        </div>
        <div>
          <div className="text-[11px] text-[var(--text-dim)] mb-0.5">Defect Type</div>
          <div className="text-sm font-medium">{vision_output.label}</div>
        </div>
        <div>
          <div className="text-[11px] text-[var(--text-dim)] mb-0.5">Verified</div>
          <div className="text-sm font-medium text-[var(--accent-green)]">
            {inspection.verification.confidence_score}/100
          </div>
        </div>
      </div>
    </div>
  );
}
