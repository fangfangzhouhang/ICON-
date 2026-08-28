from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.adapters.icon import IconCapeAdapter
from evaluation.compare_methods import build_comparison, markdown_report
from evaluation.schema import load_csv, normalize_record, validate_record, write_csv
from evaluation.validate_submission import build_report


def ok_row(sample_id: str, method: str = "icon-filter", value: float = 1.0):
    subject, rotation = sample_id.rsplit("@", 1)
    return {
        "protocol_id": "geometry-cape-v1",
        "benchmark": "cape",
        "method": method,
        "sample_id": sample_id,
        "group": "easy",
        "subject": subject,
        "rotation": int(rotation),
        "status": "ok",
        "chamfer_cm": value,
        "p2s_cm": value * 0.9,
        "normal_error": value * 0.05,
        "prediction_mesh_path": "pred.obj",
        "ground_truth_mesh_path": "gt.obj",
    }


class UnifiedSchemaTest(unittest.TestCase):
    def test_valid_success_and_explicit_failure(self):
        self.assertEqual(validate_record(ok_row("subject-a@000")), [])
        failed = normalize_record(
            {
                "method": "icon-filter",
                "sample_id": "subject-b@120",
                "status": "failed",
                "error_type": "RuntimeError",
            }
        )
        self.assertEqual(validate_record(failed), [])

    def test_success_cannot_hide_missing_metric(self):
        row = ok_row("subject-a@000")
        row["p2s_cm"] = ""
        self.assertIn("successful row missing metric: p2s_cm", validate_record(row))

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unified.csv"
            write_csv(path, [ok_row("subject-a@000")])
            loaded = load_csv(path)
            self.assertEqual(loaded[0]["sample_id"], "subject-a@000")
            self.assertAlmostEqual(loaded[0]["chamfer_cm"], 1.0)


class IconAdapterTest(unittest.TestCase):
    def test_maps_pilot_summary_without_recomputing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pilot.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "method", "group", "subject", "rotation", "status",
                        "chamfer_cm", "p2s_cm", "normal_error",
                        "prediction_mesh_path", "ground_truth_mesh_path",
                        "normal_checkpoint_sha256",
                    ],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        **ok_row("cape-case@120"),
                        "normal_checkpoint_sha256": "abc",
                    }
                )
            converted = IconCapeAdapter().convert(source)
            self.assertEqual(converted[0]["sample_id"], "cape-case@120")
            self.assertEqual(converted[0]["num_samples"], 1000)
            self.assertEqual(converted[0]["normal_checkpoint_sha256"], "abc")
            self.assertAlmostEqual(converted[0]["p2s_cm"], 0.9)


class SubmissionAndComparisonTest(unittest.TestCase):
    def test_declared_count_and_failures_are_visible(self):
        records = [
            ok_row("case-a@000"),
            normalize_record(
                {
                    "method": "icon-filter",
                    "sample_id": "case-b@000",
                    "status": "failed",
                    "error_message": "out of memory",
                }
            ),
        ]
        report = build_report(records, expected_count=2)
        self.assertTrue(report["contract_valid"])
        self.assertEqual(report["status_counts"]["failed"], 1)
        self.assertAlmostEqual(report["success_rate"], 0.5)
        self.assertFalse(build_report(records, expected_count=3)["contract_valid"])

    def test_matched_methods_get_paired_delta(self):
        icon = [ok_row("a@000", "icon", 1.0), ok_row("b@000", "icon", 2.0)]
        pifu = [ok_row("a@000", "pifu", 1.2), ok_row("b@000", "pifu", 2.2)]
        report = build_comparison({"icon": icon, "pifu": pifu})
        self.assertTrue(report["same_case_contract"])
        chamfer = next(row for row in report["paired_deltas"] if row["metric"] == "chamfer_cm")
        self.assertAlmostEqual(chamfer["mean"], 0.2)
        self.assertIn("Matched-case contract: PASS", markdown_report(report))

    def test_mismatched_cases_are_rejected_and_empty_groups_render(self):
        report = build_comparison(
            {"icon": [ok_row("a@000", "icon")], "pifu": [ok_row("b@000", "pifu")]}
        )
        self.assertFalse(report["same_case_contract"])
        self.assertIn("Matched-case contract: FAIL", markdown_report(report))
        self.assertIn("NA", markdown_report(report))

    def test_quality_means_use_shared_successful_denominator(self):
        icon = [ok_row("a@000", "icon", 1.0), ok_row("b@000", "icon", 100.0)]
        failed = normalize_record(
            {
                "method": "pifu",
                "sample_id": "b@000",
                "status": "failed",
                "error_message": "inference failed",
            }
        )
        pifu = [ok_row("a@000", "pifu", 2.0), failed]
        report = build_comparison({"icon": icon, "pifu": pifu})
        self.assertEqual(report["paired_successful_case_count"], 1)
        icon_chamfer = next(
            row
            for row in report["summary"]
            if row["method"] == "icon"
            and row["group"] == "all"
            and row["metric"] == "chamfer_cm"
        )
        self.assertEqual(icon_chamfer["count"], 1)
        self.assertAlmostEqual(icon_chamfer["mean"], 1.0)

    def test_metadata_mismatch_fails_contract(self):
        reference = ok_row("a@000", "reference")
        candidate = ok_row("a@000", "candidate")
        candidate["benchmark"] = "different-benchmark"

        report = build_comparison({"reference": [reference], "candidate": [candidate]})

        self.assertTrue(report["same_case_contract"])
        self.assertFalse(report["metadata_contract"])
        self.assertFalse(report["matched_contract"])
        self.assertEqual(len(report["metadata_mismatches"]), 1)
        self.assertIn("Matched-case contract: FAIL", markdown_report(report))


if __name__ == "__main__":
    unittest.main()
