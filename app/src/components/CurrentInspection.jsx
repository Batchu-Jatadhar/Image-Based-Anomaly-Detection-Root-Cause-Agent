import { useRef, useState } from "react";
import { ScanEye, UploadCloud, Loader2, ExternalLink, ImageOff } from "lucide-react";
import HeatmapOverlay from "./HeatmapOverlay";

const TABS = ["Original Image", "Heatmap", "Mask"];

function StatusPill({ label }) {
  const isDefect = label && label !== "Normal";
  return (
    <span
      className={`text-xs font-medium px-2.5 py-1 rounded-full ${
        isDefect
          ? "bg-[#f0526a1f] text-[var(--accent-red)] border border-[#f0526a40]"
          : "bg-[#34d3991f] text-[var(--accent-green)] border border-[#34d39940]"
      }`}
    >
      {isDefect ? "Defect Detected" : "No Defect"}
    </span>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="mb-3.5">
      <div className="text-[11px] text-[var(--text-dim)] mb-0.5">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  );
}

export default function CurrentInspection({ inspection, loading, onUpload, onViewReport }) {
  const [tab, setTab] = useState(0);
  const fileRef = useRef(null);

  const handlePick = () => fileRef.current?.click();
  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    e.target.value = "";
  };
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className="panel rounded-2xl p-6 flex-1 rise-in">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2.5">
          <ScanEye size={17} className="text-[var(--accent-cyan)]" />
          <h2 className="font-display font-semibold text-[15px]">Current Inspection</h2>
          {inspection && <StatusPill label={inspection.vision_output.label} />}
        </div>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFile}
        />
        <button
          onClick={handlePick}
          className="text-xs font-medium text-[var(--accent-violet)] hover:text-white flex items-center gap-1.5 transition-colors"
        >
          <UploadCloud size={14} />
          Upload image
        </button>
      </div>

      <div className="grid grid-cols-[1.5fr_1fr] gap-6">
        {/* Image viewer */}
        <div>
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className="relative aspect-[16/10] rounded-xl overflow-hidden border border-[var(--border-soft)] bg-[#0a0e1c]"
          >
            {loading && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-[#0a0e1cee]">
                <Loader2 size={28} className="animate-spin text-[var(--accent-violet)]" />
                <span className="text-xs text-[var(--text-muted)]">
                  Running detection &amp; root-cause pipeline…
                </span>
              </div>
            )}

            {!inspection && !loading && (
              <button
                onClick={handlePick}
                className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-[var(--text-dim)] hover:text-[var(--text-muted)] transition-colors"
              >
                <ImageOff size={26} />
                <span className="text-xs">Drop an image or click to upload</span>
              </button>
            )}

            {inspection && !loading && (
              <>
                <img
                  src={inspection.image_url}
                  alt="Inspection subject"
                  className="absolute inset-0 w-full h-full object-cover"
                />
                {tab === 1 && <HeatmapOverlay bbox={inspection.vision_output.bbox} mode="heatmap" />}
                {tab === 2 && <HeatmapOverlay bbox={inspection.vision_output.bbox} mode="mask" />}
                {tab !== 2 && (
                  <div
                    className="absolute border-2 border-[var(--accent-red)] rounded-sm"
                    style={{
                      left: `${inspection.vision_output.bbox[0] * 100}%`,
                      top: `${inspection.vision_output.bbox[1] * 100}%`,
                      width: `${inspection.vision_output.bbox[2] * 100}%`,
                      height: `${inspection.vision_output.bbox[3] * 100}%`,
                    }}
                  />
                )}
              </>
            )}
          </div>

          {/* Thumbnails / tabs */}
          <div className="grid grid-cols-3 gap-3 mt-3">
            {TABS.map((label, i) => (
              <button
                key={label}
                onClick={() => setTab(i)}
                disabled={!inspection}
                className={`rounded-lg overflow-hidden border transition-colors disabled:opacity-40 ${
                  tab === i ? "border-[var(--accent-violet)]" : "border-[var(--border-soft)]"
                }`}
              >
                <div className="relative aspect-video bg-[#0a0e1c]">
                  {inspection && (
                    <>
                      <img
                        src={inspection.image_url}
                        className="absolute inset-0 w-full h-full object-cover"
                      />
                      {i === 1 && <HeatmapOverlay bbox={inspection.vision_output.bbox} mode="heatmap" />}
                      {i === 2 && <HeatmapOverlay bbox={inspection.vision_output.bbox} mode="mask" />}
                    </>
                  )}
                </div>
                <div className="text-[11px] text-center py-1.5 text-[var(--text-muted)]">
                  {label}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Details */}
        <div className="flex flex-col">
          {inspection ? (
            <>
              <DetailRow label="Inspection ID" value={inspection.inspection_id} />
              <DetailRow label="Component" value={inspection.component} />
              <DetailRow label="Line" value={inspection.line} />
              <DetailRow
                label="Timestamp"
                value={new Date(inspection.timestamp).toLocaleString("en-US", {
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                })}
              />
              <DetailRow label="AI Model" value={inspection.model} />
              <DetailRow
                label="Confidence"
                value={
                  <span className="text-[var(--accent-green)]">
                    {(inspection.vision_output.confidence * 100).toFixed(1)}%
                  </span>
                }
              />

              <button
                onClick={onViewReport}
                className="mt-auto w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-[var(--accent-violet-dim)] border border-[#8b6cf84d] text-sm font-medium hover:bg-[#8b6cf83a] transition-colors"
              >
                View Full Report
                <ExternalLink size={14} />
              </button>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-center text-sm text-[var(--text-dim)] px-4">
              Upload an industrial image to run defect detection, root-cause analysis, and get a
              maintenance recommendation.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
