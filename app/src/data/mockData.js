// Mock data shaped EXACTLY like the contracts Person 1 (vision) and Person 2 (RAG/agents)
// promised in the README. When their modules land, only src/api/pipeline.js needs to change —
// every component below just consumes this shape.

export const DEFECT_TYPES = [
  "Crack",
  "Scratch",
  "Dent",
  "Corrosion",
  "Burn Mark",
  "Surface Defect",
  "Hole",
];

export const ROOT_CAUSES = {
  Crack: {
    cause:
      "High residual stress from an uneven cooling rate during heat treatment, likely compounded by cyclic loading on the component.",
    actions: [
      "Verify furnace cooling curve against spec",
      "Inspect heat-treatment fixture alignment",
      "Check for material batch inconsistencies",
    ],
    severity: "High",
    priority: "High",
    downtime: "30 – 45 min",
  },
  Corrosion: {
    cause:
      "Moisture ingress at a coating breach, accelerated by inconsistent ambient humidity control on the line.",
    actions: [
      "Inspect protective coating integrity",
      "Check enclosure seals for the affected station",
      "Review humidity logs for the last 72 hours",
    ],
    severity: "Medium",
    priority: "Medium",
    downtime: "15 – 20 min",
  },
  Scratch: {
    cause:
      "Surface contact during transfer between Line 2 and Line 3, consistent with a misaligned conveyor guide.",
    actions: [
      "Inspect conveyor guide alignment",
      "Check handling fixture for burrs",
      "Review operator handling procedure",
    ],
    severity: "Low",
    priority: "Low",
    downtime: "5 – 10 min",
  },
  Dent: {
    cause:
      "Impact damage during transport, likely from insufficient cushioning in the staging rack.",
    actions: [
      "Inspect staging rack padding",
      "Review transport handling checklist",
      "Flag batch for dimensional re-check",
    ],
    severity: "Medium",
    priority: "Medium",
    downtime: "10 – 15 min",
  },
};

function pad(n) {
  return String(n).padStart(2, "0");
}

let inspectionCounter = 124;

// Mirrors src/inference.py -> predict()
function mockPredict() {
  const defect = DEFECT_TYPES[Math.floor(Math.random() * 3)]; // bias toward first 3 for demo coherence
  const confidence = +(0.88 + Math.random() * 0.11).toFixed(3);
  return {
    label: defect,
    confidence,
    bbox: [
      +(0.3 + Math.random() * 0.2).toFixed(2),
      +(0.25 + Math.random() * 0.2).toFixed(2),
      +(0.15 + Math.random() * 0.1).toFixed(2),
      +(0.12 + Math.random() * 0.1).toFixed(2),
    ],
    heatmap_overlay_path: "/mock/heatmap_overlay.png",
  };
}

// Mirrors src/agents/orchestrator.py -> run_pipeline()
export function mockRunPipeline(imageUrl) {
  inspectionCounter += 1;
  const vision = mockPredict();
  const rc = ROOT_CAUSES[vision.label] || ROOT_CAUSES.Crack;
  const now = new Date();

  return {
    inspection_id: `INS-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(
      now.getDate()
    )}-${String(inspectionCounter).padStart(5, "0")}`,
    component: ["Metal Shaft", "Gear Housing", "Bearing Block", "Weld Joint"][
      Math.floor(Math.random() * 4)
    ],
    line: ["Assembly Line 1", "Assembly Line 2", "Assembly Line 3"][
      Math.floor(Math.random() * 3)
    ],
    timestamp: now.toISOString(),
    model: "YOLOv11 + ResNet50",
    image_url: imageUrl,
    vision_output: vision,
    findings: {
      summary: `${vision.label} detected with ${(vision.confidence * 100).toFixed(
        1
      )}% calibrated confidence. Localized region flagged by Grad-CAM aligns with predicted bounding box (IoU 0.81).`,
    },
    report: {
      impression: `${vision.label} identified on the inspected surface. Defect area approx. ${Math.round(
        800 + Math.random() * 900
      )} px². Confidence calibrated via temperature scaling.`,
      root_cause: rc.cause,
      supporting_evidence: [
        "3 similar historical cases retrieved via FAISS (cosine sim > 0.86)",
        "Matches failure pattern in internal maintenance guideline §4.2",
      ],
      recommended_next_steps: rc.actions,
    },
    verification: {
      confidence_score: Math.round(90 + Math.random() * 9),
      flagged_claims: [],
      verified: true,
    },
    risk: {
      severity: rc.severity,
      priority: rc.priority,
      downtime: rc.downtime,
      failure_probability: Math.round(60 + Math.random() * 30),
    },
    status: "pending_review", // pending_review | approved | revised
  };
}

export const kpis = [
  { label: "Inspections Today", value: "128", delta: "+12% vs yesterday", up: true, icon: "scan" },
  { label: "Defects Detected", value: "23", delta: "+8% vs yesterday", up: true, icon: "alert" },
  { label: "Defect Rate", value: "18.0%", delta: "-3% vs yesterday", up: false, icon: "percent" },
  { label: "Avg. Confidence", value: "94.2%", delta: "+2% vs yesterday", up: true, icon: "bar" },
];

export const historicalCases = [
  { id: "INS-20260720-00098", label: "Surface Crack", line: "Line 3", age: "2 days ago", status: "Resolved" },
  { id: "INS-20260718-00081", label: "Surface Crack", line: "Line 1", age: "4 days ago", status: "Resolved" },
  { id: "INS-20260716-00073", label: "Surface Crack", line: "Line 2", age: "12 days ago", status: "In Progress" },
];

export const modelMetrics = [
  { label: "mAP@0.5", value: "0.892" },
  { label: "Precision", value: "0.925" },
  { label: "Recall", value: "0.876" },
  { label: "F1 Score", value: "0.899" },
  { label: "Avg. Inference", value: "41 ms" },
];
