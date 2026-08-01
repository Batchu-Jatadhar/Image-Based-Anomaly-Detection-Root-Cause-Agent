import { Clock3, ArrowRight } from "lucide-react";

const STATUS_STYLE = {
  Resolved: "bg-[#34d3991f] text-[var(--accent-green)] border-[#34d39940]",
  "In Progress": "bg-[#f5a5241f] text-[var(--accent-amber)] border-[#f5a52440]",
};

export default function HistoricalCases({ cases }) {
  return (
    <div className="panel rounded-2xl p-6 rise-in">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <Clock3 size={17} className="text-[var(--accent-cyan)]" />
          <h2 className="font-display font-semibold text-[15px]">Similar Historical Cases</h2>
        </div>
        <button className="text-xs font-medium text-[var(--accent-violet)] hover:text-white flex items-center gap-1 transition-colors">
          View All <ArrowRight size={13} />
        </button>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {cases.map((c) => (
          <div
            key={c.id}
            className="rounded-xl border border-[var(--border-soft)] p-3 hover:border-[#8b6cf84d] transition-colors cursor-pointer"
          >
            <div className="aspect-video rounded-lg mb-3 bg-gradient-to-br from-[#1a2040] to-[#0d1122] flex items-center justify-center overflow-hidden relative">
              <div
                className="absolute w-16 h-16 rounded-full"
                style={{
                  background:
                    "radial-gradient(circle, rgba(240,20,20,0.8) 0%, rgba(255,180,20,0.6) 40%, rgba(20,120,255,0.3) 75%, transparent 100%)",
                  mixBlendMode: "screen",
                }}
              />
            </div>
            <div className="text-[11px] text-[var(--text-dim)] mb-1 font-mono truncate">
              Case #{c.id}
            </div>
            <div className="text-sm font-medium mb-2">{c.label}</div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-[var(--text-muted)]">
                {c.line} · {c.age}
              </span>
              <span
                className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${STATUS_STYLE[c.status]}`}
              >
                {c.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
