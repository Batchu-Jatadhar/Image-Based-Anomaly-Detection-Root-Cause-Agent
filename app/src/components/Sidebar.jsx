import {
  LayoutDashboard,
  ScanSearch,
  History,
  BookOpen,
  Bell,
  FileBarChart,
  Settings,
  Boxes,
} from "lucide-react";

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "new", label: "New Inspection", icon: ScanSearch },
  { id: "history", label: "History", icon: History },
  { id: "knowledge", label: "Knowledge Base", icon: BookOpen },
  { id: "alerts", label: "Alerts", icon: Bell },
  { id: "reports", label: "Reports", icon: FileBarChart },
  { id: "settings", label: "Settings", icon: Settings },
];

export default function Sidebar({ active, onNavigate }) {
  return (
    <aside className="w-64 shrink-0 h-full flex flex-col justify-between border-r border-[#1a2038] px-4 py-6">
      <div>
        <div className="flex items-center gap-3 px-2 mb-8">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#8b6cf8] to-[#4f6df5] flex items-center justify-center panel-glow">
            <Boxes size={20} className="text-white" strokeWidth={2.25} />
          </div>
          <div>
            <div className="font-display font-bold text-lg leading-none tracking-tight">
              CruxAI
            </div>
            <div className="text-[11px] text-[var(--text-dim)] mt-1 tracking-wide">
              AI Manufacturing Assistant
            </div>
          </div>
        </div>

        <nav className="flex flex-col gap-1">
          {NAV.map((item) => {
            const Icon = item.icon;
            const isActive = active === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors text-left ${
                  isActive
                    ? "bg-[var(--accent-violet-dim)] text-white border border-[#8b6cf84d]"
                    : "text-[var(--text-muted)] hover:text-white hover:bg-white/[0.03] border border-transparent"
                }`}
              >
                <Icon size={17} strokeWidth={2} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      <div className="flex flex-col gap-3">
        <div className="panel rounded-2xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-[var(--text-muted)]">System Status</span>
            <span className="w-2 h-2 rounded-full bg-[var(--accent-green)] pulse-dot" />
          </div>
          <div className="text-sm font-medium">All Systems Operational</div>
        </div>

        <div className="flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-white/[0.03] cursor-pointer">
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-[#35d3e8] to-[#8b6cf8] flex items-center justify-center text-xs font-semibold text-[#0a0d18]">
            DA
          </div>
          <div className="leading-tight">
            <div className="text-sm font-medium">Dia</div>
            <div className="text-[11px] text-[var(--text-dim)]">Full-Stack &amp; Systems</div>
          </div>
        </div>
      </div>
    </aside>
  );
}
