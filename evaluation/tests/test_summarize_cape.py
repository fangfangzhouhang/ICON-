import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.summarize_cape import build_summary, load_rows, percentile, write_outputs


class CapeSummaryTest(unittest.TestCase):
    def write_fixture(self, root: Path) -> Path:
        path = root / "pilot_summary.csv"
        fieldnames = [
            "status",
            "group",
            "subject_index",
            "subject",
            "rotation",
            "dataset_index",
            "mcube_res",
            "seed",
            "chamfer_cm",
            "p2s_cm",
            "normal_error",
            "runtime_seconds",
            "peak_memory_allocated_gib",
            "peak_memory_reserved_gib",
            "error_type",
            "error_message",
        ]
        rows = [
            ["ok", "easy", 0, "a", 0, 0, 256, 1993, 1, 2, 0.1, 10, 8, 12, "", ""],
            ["ok", "easy", 0, "a", 120, 1, 256, 1993, 2, 3, 0.2, 2, 3, 4, "", ""],
            ["ok", "hard", 50, "b", 0, 150, 256, 1993, 3, 4, 0.3, 4, 5, 6, "", ""],
            ["failed", "hard", 50, "b", 120, 151, 256, 1993, "", "", "", "", "", "", "RuntimeError", "boom"],
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fieldnames)
            writer.writerows(rows)
        return path

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 0.9), 1.9)

    def test_summary_counts_failures_and_excludes_first_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.write_fixture(Path(temp))
            summary = build_summary(load_rows(source), source)
            self.assertEqual(summary["case_count"], 4)
            self.assertEqual(summary["success_count"], 3)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["unique_subject_count"], 2)
            self.assertAlmostEqual(summary["overall"]["p2s_cm"]["mean"], 3.0)
            steady = summary["steady_state_system_metrics"]["statistics"]
            self.assertAlmostEqual(steady["runtime_seconds"]["mean"], 3.0)
            self.assertEqual(summary["quality_extremes"]["chamfer_cm"]["best"]["subject"], "a")
            self.assertEqual(summary["quality_extremes"]["chamfer_cm"]["worst"]["subject"], "b")

    def test_writes_all_three_report_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.write_fixture(root)
            summary = build_summary(load_rows(source), source)
            output = root / "analysis"
            write_outputs(summary, output)
            self.assertTrue((output / "benchmark_summary.csv").is_file())
            self.assertTrue((output / "benchmark_report.md").is_file())
            payload = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_count"], 1)

    def test_explicit_warmup_markers_cover_resumed_sessions(self):
        rows = [
            {
                "status": "ok",
                "subject": "a",
                "group": "easy",
                "rotation": 0,
                "warmup_affected": "True",
                "runtime_seconds": 10.0,
                "chamfer_cm": 1.0,
                "p2s_cm": 1.0,
                "normal_error": 0.1,
                "peak_memory_allocated_gib": 8.0,
                "peak_memory_reserved_gib": 10.0,
            },
            {
                "status": "ok",
                "subject": "b",
                "group": "hard",
                "rotation": 0,
                "warmup_affected": "True",
                "runtime_seconds": 20.0,
                "chamfer_cm": 1.0,
                "p2s_cm": 1.0,
                "normal_error": 0.1,
                "peak_memory_allocated_gib": 8.0,
                "peak_memory_reserved_gib": 10.0,
            },
            {
                "status": "ok",
                "subject": "c",
                "group": "hard",
                "rotation": 120,
                "warmup_affected": "False",
                "runtime_seconds": 3.0,
                "chamfer_cm": 1.0,
                "p2s_cm": 1.0,
                "normal_error": 0.1,
                "peak_memory_allocated_gib": 3.0,
                "peak_memory_reserved_gib": 4.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.csv"
            source.write_text("fixture", encoding="utf-8")
            summary = build_summary(rows, source)
        steady = summary["steady_state_system_metrics"]
        self.assertEqual(len(steady["excluded_cases"]), 2)
        self.assertEqual(steady["statistics"]["runtime_seconds"]["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
