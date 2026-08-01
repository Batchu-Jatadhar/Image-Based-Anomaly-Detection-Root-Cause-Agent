import { Calendar, Bell, UploadCloud } from "lucide-react";

export default function TopBar({ title, subtitle, onNewInspection, alertCount = 3 }) {
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  const timeStr = now.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });

  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h1 className="font-display text-2xl font-bold tracking-tight">{title}</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">{subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onNewInspection}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-br from-[#8b6cf8] to-[#6a4ef0] text-sm font-medium hover:opacity-90 transition-opacity panel-glow"
        >
          <UploadCloud size={16} />
          New Inspection
        </button>
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl panel text-sm text-[var(--text-muted)]">
          <Calendar size={15} />
          {dateStr} · {timeStr}
        </div>
        <button className="relative w-10 h-10 rounded-xl panel flex items-center justify-center hover:border-[#8b6cf866] transition-colors">
          <Bell size={16} className="text-[var(--text-muted)]" />
          {alertCount > 0 && (
            <span className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-[var(--accent-red)] text-[10px] font-semibold flex items-center justify-center">
              {alertCount}
            </span>
          )}
        </button>
      </div>
    </div>
  );
}
