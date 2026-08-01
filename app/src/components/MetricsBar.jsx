import { Target, Crosshair, Radar, Gauge, Timer } from "lucide-react";
import { LineChart, Line, ResponsiveContainer } from "recharts";

const ICONS = {
  "mAP@0.5": Target,
  Precision: Crosshair,
  Recall: Radar,
  "F1 Score": Gauge,
  "Avg. Inference": Timer,
};

function spark(seed) {
  const data = [];
  let v = 50 + seed;
  for (let i = 0; i < 10; i++) {
    v += (Math.sin(i + seed) * 8) + (Math.random() * 4 - 2);
    data.push({ v });
  }
  return data;
}

export default function MetricsBar({ metrics }) {
  return (
    <div className="panel rounded-2xl px-6 py-4 grid grid-cols-5 gap-4 rise-in">
      {metrics.map((m, i) => {
        const Icon = ICONS[m.label];
        return (
          <div key={m.label} className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-white/[0.04] flex items-center justify-center shrink-0">
              <Icon size={15} className="text-[var(--text-muted)]" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] text-[var(--text-dim)] truncate">{m.label}</div>
              <div className="text-sm font-semibold">{m.value}</div>
            </div>
            <div className="w-16 h-8 shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={spark(i * 3)}>
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke="#35d3e8"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}
