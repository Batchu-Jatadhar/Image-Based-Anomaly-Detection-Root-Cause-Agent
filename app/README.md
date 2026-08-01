# CruxAI Dashboard (Frontend)

Person 3's chunk of the Image-Based Anomaly Detection & Root-Cause Agent project —
the human-facing dashboard for uploading inspection images, viewing AI diagnosis
(defect + Grad-CAM localization), root cause & recommended action, and the
human-in-the-loop Approve/Revise gate.

Currently wired to **mock data** that mirrors the `predict()` / `run_pipeline()`
contracts from the project README, so the whole UI works standalone right now.

## Run it

```bash
npm install
npm run dev
```

## Connecting the real backend

Everything mock-related lives in two files:

- `src/data/mockData.js` — shape of the pipeline output (matches `run_pipeline()`)
- `src/api/pipeline.js` — the actual "API call". Replace the body of `runPipeline()`
  with a real `fetch()` to the FastAPI endpoint once Person 1/2's backend is up.
  See the `TODO(backend)` comment inside that file for the exact snippet.

No component needs to change — they all just consume the object shape already
defined by the contract.

## Structure

```
src/
  api/pipeline.js          swap point for real backend
  data/mockData.js          mock pipeline responses + KPIs + historical cases
  components/
    Sidebar.jsx
    TopBar.jsx
    StatCards.jsx
    CurrentInspection.jsx   image upload, tabs (original/heatmap/mask), bbox
    HeatmapOverlay.jsx      CSS Grad-CAM stand-in, positioned from bbox
    ConfidenceGauge.jsx     radial confidence ring
    AIDiagnosis.jsx
    RootCause.jsx
    HistoricalCases.jsx
    MetricsBar.jsx          mAP / precision / recall / F1 / inference time
    ReportModal.jsx         full report + Approve/Revise gate + JSON/PDF export
  App.jsx
```

## What's real vs. mocked

- **Real**: upload flow, state management, JSON export, print-to-PDF, the
  Approve/Revise review gate (writes to mock functions that mimic the real
  API calls you'll add).
- **Mocked**: the actual defect prediction and Grad-CAM heatmap — `HeatmapOverlay.jsx`
  draws a CSS gradient positioned using the bbox coordinates rather than a real
  heatmap PNG. Swap in `vision_output.heatmap_overlay_path` once it exists.
