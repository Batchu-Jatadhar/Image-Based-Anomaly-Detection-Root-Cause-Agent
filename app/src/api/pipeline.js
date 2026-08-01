import { mockRunPipeline } from "../data/mockData";

/**
 * runPipeline(imageFile)
 *
 * TODO(backend): once Person 2's FastAPI endpoint is live, replace the body of
 * this function with something like:
 *
 *   const form = new FormData();
 *   form.append("image", imageFile);
 *   const res = await fetch(`${import.meta.env.VITE_API_BASE}/run_pipeline`, {
 *     method: "POST",
 *     body: form,
 *   });
 *   if (!res.ok) throw new Error("Pipeline request failed");
 *   return res.json();
 *
 * The response shape is already defined by the orchestrator.py contract in the
 * README, and mockRunPipeline() mirrors it field-for-field, so no component
 * changes should be needed elsewhere.
 */
export async function runPipeline(imageUrl) {
  await new Promise((r) => setTimeout(r, 1400 + Math.random() * 800));
  return mockRunPipeline(imageUrl);
}

export async function approveInspection(inspectionId, reviewerName) {
  await new Promise((r) => setTimeout(r, 500));
  return {
    inspection_id: inspectionId,
    status: "approved",
    reviewer: reviewerName,
    reviewed_at: new Date().toISOString(),
  };
}

export async function reviseInspection(inspectionId, revisedFields) {
  await new Promise((r) => setTimeout(r, 500));
  return {
    inspection_id: inspectionId,
    status: "revised",
    revised_fields: revisedFields,
    reviewed_at: new Date().toISOString(),
  };
}
