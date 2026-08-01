import { ScanSearch, AlertTriangle, Percent, BarChart3, ArrowUp, ArrowDown } from "lucide-react";

const ICONS = { scan: ScanSearch, alert: AlertTriangle, percent: Percent, bar: BarChart3 };
const TINTS = {
  scan: "from-[#35d3e8]/15 text-[#35d3e8]",
  alert: "from-[#8b6cf8]/15 text-[#8b6cf8]",
  percent: "from-[#34d399]/15 text-[#34d399]",
  bar: "from-[#f5a524]/15 text-[#f5a524]",
};

export default function StatCards({ kpis }) {
  return (
    <div className="grid grid-cols-4 gap-4 mb-6">
      {kpis.map((k) => {
        const Icon = ICONS[k.icon];
        return (
          <div key={k.label} className="panel rounded-2xl p-5 rise-in">
            <div className="flex items-center justify-between mb-4">
              <div
                className={`w-9 h-9 rounded-lg bg-gradient-to-br ${TINTS[k.icon]} flex items-center justify-center`}
              >
                <Icon size={17} strokeWidth={2.25} />
              </div>
            </div>
            <div className="text-xs text-[var(--text-muted)] mb-1">{k.label}</div>
            <div className="font-display text-2xl font-bold mb-1.5">{k.value}</div>
            <div
              className={`flex items-center gap-1 text-xs font-medium ${
                k.up ? "text-[var(--accent-green)]" : "text-[var(--accent-red)]"
              }`}
            >
              {k.up ? <ArrowUp size={12} /> : <ArrowDown size={12} />}
              {k.delta}
            </div>
          </div>
        );
      })}
    </div>
  );
}
