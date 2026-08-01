// Renders a pseudo-Grad-CAM overlay positioned from the normalized [x,y,w,h] bbox
// returned by predict(). Once Person 1 delivers real heatmap_overlay_path PNGs,
// swap this for an <img src={heatmap_overlay_path} /> — the bbox-driven layout
// here exists purely so the dashboard has something honest to show against mock data.
export default function HeatmapOverlay({ bbox, mode = "heatmap" }) {
  const [x, y, w, h] = bbox;
  const cx = (x + w / 2) * 100;
  const cy = (y + h / 2) * 100;
  const spread = Math.max(w, h) * 140;

  if (mode === "mask") {
    return (
      <div className="absolute inset-0 bg-black">
        <div
          className="absolute rounded-full bg-white"
          style={{
            left: `${cx}%`,
            top: `${cy}%`,
            width: `${spread}%`,
            height: `${spread}%`,
            transform: "translate(-50%, -50%)",
            filter: "blur(2px)",
          }}
        />
      </div>
    );
  }

  return (
    <div
      className="absolute inset-0"
      style={{
        background: `radial-gradient(circle at ${cx}% ${cy}%, rgba(240,20,20,0.85) 0%, rgba(255,150,20,0.75) ${
          spread * 0.35
        }%, rgba(255,230,20,0.5) ${spread * 0.6}%, rgba(20,120,255,0.35) ${spread * 0.95}%, transparent ${
          spread * 1.3
        }%)`,
        mixBlendMode: "screen",
      }}
    />
  );
}
