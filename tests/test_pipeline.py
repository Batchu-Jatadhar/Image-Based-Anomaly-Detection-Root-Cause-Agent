import os
import sys
import unittest

# Ensure the workspace directory is in the path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import run_pipeline

class TestAnomalyPipeline(unittest.TestCase):
    """Integration and schema contract verification tests for Multi-Agent AI Pipeline."""

    def test_pipeline_normal_case(self):
        """Verifies the pipeline executes correctly with standard inputs and produces expected schema."""
        image_path = "data/images/component_normal.png"
        patient_meta = {"machine_id": "M_405", "operator": "E_12"}
        
        result = run_pipeline(image_path, patient_meta)
        
        # Verify root schema
        self.assertIn("vision_output", result)
        self.assertIn("findings", result)
        self.assertIn("report", result)
        self.assertIn("verification", result)

        # Verify vision_output block schema
        vo = result["vision_output"]
        self.assertEqual(vo["label"], "Normal")
        self.assertIsInstance(vo["confidence"], float)
        self.assertTrue(0.0 <= vo["confidence"] <= 1.0)
        self.assertIsNone(vo["bbox"])
        self.assertIsInstance(vo["heatmap_overlay_path"], str)

        # Verify findings block schema
        findings = result["findings"]
        self.assertIn("summary", findings)
        self.assertIsInstance(findings["summary"], str)
        self.assertIn("Normal", findings["summary"])

        # Verify report block schema
        report = result["report"]
        self.assertIn("impression", report)
        self.assertIn("root_cause", report)
        self.assertIn("supporting_evidence", report)
        self.assertIn("recommended_next_steps", report)
        self.assertIsInstance(report["impression"], str)
        self.assertIsInstance(report["root_cause"], str)
        self.assertIsInstance(report["supporting_evidence"], list)
        self.assertIsInstance(report["recommended_next_steps"], list)

        # Verify verification block schema
        verification = result["verification"]
        self.assertIn("confidence_score", verification)
        self.assertIn("flagged_claims", verification)
        self.assertIn("verified", verification)
        self.assertIsInstance(verification["confidence_score"], float)
        self.assertTrue(0.0 <= verification["confidence_score"] <= 1.0)
        self.assertIsInstance(verification["flagged_claims"], list)
        self.assertIsInstance(verification["verified"], bool)

    def test_pipeline_defect_case(self):
        """Verifies the pipeline processes defects, normalizes values, and structures reports correctly."""
        image_path = "data/images/component_crack.png"
        patient_meta = {"machine_id": "M_405"}
        
        result = run_pipeline(image_path, patient_meta)

        vo = result["vision_output"]
        self.assertEqual(vo["label"], "Crack")
        self.assertListEqual(vo["bbox"], [120, 80, 45, 55])
        
        findings = result["findings"]
        self.assertIn("Crack", findings["summary"])

        # Check diagnostic report fields have content
        report = result["report"]
        self.assertTrue(len(report["impression"]) > 0)
        self.assertTrue(len(report["root_cause"]) > 0)
        self.assertTrue(len(report["supporting_evidence"]) > 0)
        self.assertTrue(len(report["recommended_next_steps"]) > 0)

        # Check verification fields
        verification = result["verification"]
        self.assertIsInstance(verification["verified"], bool)

if __name__ == "__main__":
    unittest.main()
