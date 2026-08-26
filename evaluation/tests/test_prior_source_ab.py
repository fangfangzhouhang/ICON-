import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.run_prior_source_ab import evaluate_cases, prepare_inputs, run_inference, select_cases
from evaluation.summarize_prior_source_ab import build_summary, load_rows


class PriorSourceABTest(unittest.TestCase):
    def test_case_selection_requires_complete_pair_contract(self):
        rows = [
            {
                "status": "ok",
                "group": "easy",
                "subject_index": "0",
                "subject": "easy-person",
                "rotation": "0",
                "prediction_mesh_path": "a.obj",
                "ground_truth_mesh_path": "gt-a.obj",
            },
            {
                "status": "ok",
                "group": "hard",
                "subject_index": "50",
                "subject": "hard-person",
                "rotation": "0",
                "prediction_mesh_path": "b.obj",
                "ground_truth_mesh_path": "gt-b.obj",
            },
        ]
        selected = select_cases(rows, [0, 50], [0])
        self.assertEqual([case["sample_name"] for case in selected], ["ab-s000-r000", "ab-s050-r000"])
        with self.assertRaisesRegex(ValueError, "Expected 4"):
            select_cases(rows, [0, 50], [0, 120])

    def test_summary_uses_paired_delta_and_counts_failures(self):
        rows = [
            {
                "status": "ok",
                "group": "easy",
                "subject": "a",
                "rotation": 0,
                "prepared_chamfer_cm": 1.0,
                "pixie_chamfer_cm": 1.2,
                "delta_chamfer_cm": 0.2,
                "prepared_p2s_cm": 0.9,
                "pixie_p2s_cm": 1.0,
                "delta_p2s_cm": 0.1,
                "prepared_normal_error": 0.05,
                "pixie_normal_error": 0.06,
                "delta_normal_error": 0.01,
            },
            {
                "status": "ok",
                "group": "hard",
                "subject": "b",
                "rotation": 0,
                "prepared_chamfer_cm": 1.0,
                "pixie_chamfer_cm": 0.8,
                "delta_chamfer_cm": -0.2,
                "prepared_p2s_cm": 0.9,
                "pixie_p2s_cm": 0.8,
                "delta_p2s_cm": -0.1,
                "prepared_normal_error": 0.05,
                "pixie_normal_error": 0.04,
                "delta_normal_error": -0.01,
            },
        ]
        summary = build_summary(rows, bootstrap_samples=100, seed=7)
        overall = summary["slices"][0]
        self.assertEqual(overall["success_count"], 2)
        self.assertAlmostEqual(
            overall["metrics"]["chamfer_cm"]["delta_mean_pixie_minus_prepared"],
            0.0,
        )
        self.assertAlmostEqual(
            overall["metrics"]["chamfer_cm"]["pixie_worse_fraction"],
            0.5,
        )

    def test_csv_loader_rejects_missing_metric_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["status", "group", "subject", "rotation"])
                writer.writeheader()
                writer.writerow({"status": "ok", "group": "easy", "subject": "a", "rotation": 0})
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_rows(path)

    def test_prepare_inputs_removes_stale_generated_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data" / "cape_3views" / "person" / "render"
            source.mkdir(parents=True)
            (source / "000.png").write_bytes(b"current")
            output = root / "output"
            input_dir = output / "inputs"
            input_dir.mkdir(parents=True)
            (input_dir / "ab-s999-r000.png").write_bytes(b"stale")
            (input_dir / "notes.txt").write_text("keep", encoding="utf-8")
            cases = [
                {
                    "sample_name": "ab-s000-r000",
                    "group": "easy",
                    "subject_index": 0,
                    "subject": "person",
                    "rotation": 0,
                    "prediction_mesh_path": "pred.obj",
                    "ground_truth_mesh_path": "gt.obj",
                }
            ]
            result = prepare_inputs(root, output, cases)
            self.assertFalse((result / "ab-s999-r000.png").exists())
            self.assertEqual((result / "ab-s000-r000.png").read_bytes(), b"current")
            self.assertTrue((result / "notes.txt").is_file())

    @patch("evaluation.run_prior_source_ab.subprocess.run")
    def test_inference_removes_stale_mesh_and_trace_before_rerun(self, run_mock):
        run_mock.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "out" / "inputs"
            input_dir.mkdir(parents=True)
            (input_dir / "ab-s000-r000.png").write_bytes(b"image")
            generated = root / "out" / "pixie-inference" / "icon-filter" / "obj"
            generated.mkdir(parents=True)
            stale_mesh = generated / "ab-s000-r000_recon.obj"
            stale_trace = generated / "ab-s000-r000_trace.json"
            stale_mesh.write_text("stale", encoding="utf-8")
            stale_trace.write_text("stale", encoding="utf-8")
            keep = generated / "other-sample_recon.obj"
            keep.write_text("keep", encoding="utf-8")

            run_inference(root, root / "out", input_dir, root / "config.yaml", 0, 100)

            self.assertFalse(stale_mesh.exists())
            self.assertFalse(stale_trace.exists())
            self.assertTrue(keep.exists())
            run_mock.assert_called_once()

    def test_failed_case_is_written_into_the_scientific_denominator(self):
        fake_metrics = types.SimpleNamespace(evaluate_aligned_meshes=lambda *args, **kwargs: {})
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"evaluation.metrics.official_icon": fake_metrics}
        ):
            output = Path(temp)
            case = {
                "sample_name": "ab-s000-r000",
                "group": "easy",
                "subject_index": 0,
                "subject": "person",
                "rotation": 0,
                "prediction_mesh_path": str(output / "prepared.obj"),
                "ground_truth_mesh_path": str(output / "gt.obj"),
            }
            records = evaluate_cases(output, [case], "cpu", 100, 7, 1.001)
            self.assertEqual(records[0]["status"], "failed")
            self.assertEqual(records[0]["error_type"], "RuntimeError")
            self.assertIn("Expected one", records[0]["error_message"])
            self.assertEqual(records[0]["prepared_chamfer_cm"], "")
            with (output / "paired_summary.csv").open(newline="", encoding="utf-8") as stream:
                written = list(csv.DictReader(stream))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
