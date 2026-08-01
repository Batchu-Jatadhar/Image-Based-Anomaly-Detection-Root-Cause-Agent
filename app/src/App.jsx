import { useState, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import StatCards from "./components/StatCards";
import CurrentInspection from "./components/CurrentInspection";
import AIDiagnosis from "./components/AIDiagnosis";
import RootCause from "./components/RootCause";
import HistoricalCases from "./components/HistoricalCases";
import MetricsBar from "./components/MetricsBar";
import ReportModal from "./components/ReportModal";
import { runPipeline } from "./api/pipeline";
import { kpis, historicalCases, modelMetrics } from "./data/mockData";

export default function App() {
  const [nav, setNav] = useState("dashboard");
  const [inspection, setInspection] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [fileRefTrigger, setFileRefTrigger] = useState(0);

  const handleUpload = useCallback(async (file) => {
    const imageUrl = URL.createObjectURL(file);
    setLoading(true);
    try {
      const result = await runPipeline(imageUrl);
      setInspection(result);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleUpdateInspection = (updated) => setInspection(updated);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar active={nav} onNavigate={setNav} />

      <main className="flex-1 overflow-y-auto px-8 py-6">
        <TopBar
          title="Inspection Overview"
          subtitle="Real-time AI powered defect detection and analysis"
          onNewInspection={() => setFileRefTrigger((n) => n + 1)}
        />

        <StatCards kpis={kpis} />

        <div className="grid grid-cols-[1.6fr_1fr] gap-5 mb-5 items-start">
          <CurrentInspection
            key={fileRefTrigger}
            inspection={inspection}
            loading={loading}
            onUpload={handleUpload}
            onViewReport={() => inspection && setShowReport(true)}
          />
          <div className="flex flex-col gap-5">
            <AIDiagnosis inspection={inspection} />
            <RootCause inspection={inspection} onOpenReport={() => inspection && setShowReport(true)} />
          </div>
        </div>

        <div className="mb-5">
          <HistoricalCases cases={historicalCases} />
        </div>

        <MetricsBar metrics={modelMetrics} />
      </main>

      {showReport && inspection && (
        <ReportModal
          inspection={inspection}
          onClose={() => setShowReport(false)}
          onUpdate={handleUpdateInspection}
        />
      )}
    </div>
  );
}
