import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.compare_official_parity import build_report, load_official, write_outputs


class OfficialParityTest(unittest.TestCase):
    def make_fixture(self, root: Path, hard_offset: float = 0.0):
        csv_path = root / "pilot_summary.csv"
        fields = [
            "status",
            "group",
            "subject",
            "rotation",
            "chamfer_cm",
            "p2s_cm",
            "normal_error",
        ]
        rows = []
        for subject_index in range(150):
            group = "easy" if subject_index < 50 else "hard"
            base = 0.8 if group == "easy" else 0.9
            for rotation in (0, 120, 240):
                rows.append(
                    {
                        "status": "ok",
                        "group": group,
                        "subject": f"subject-{subject_index:03d}",
                        "rotation": rotation,
                        "chamfer_cm": base,
                        "p2s_cm": base - 0.03,
                        "normal_error": 0.05,
                    }
                )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        official_path = root / "test_results.npy"
        np.save(
            official_path,
            {
                "cape-easy-chamfer": 0.8,
                "cape-easy-p2s": 0.77,
                "cape-easy-NC": 0.05,
                "cape-hard-chamfer": 0.9 + hard_offset,
                "cape-hard-p2s": 0.87,
                "cape-hard-NC": 0.05,
            },
            allow_pickle=True,
        )
        return official_path, csv_path

    def args(self, root: Path, official_path: Path, csv_path: Path):
        return argparse.Namespace(
            official_npy=official_path,
            custom_csv=csv_path,
            output_dir=root / "out",
            surface_abs_tol_cm=0.03,
            normal_abs_tol=0.003,
            relative_tol=0.03,
        )

    def test_complete_matching_contract_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            official_path, csv_path = self.make_fixture(root)
            args = self.args(root, official_path, csv_path)
            report = build_report(args)
            self.assertEqual(report["verdict"], "PASS")
            self.assertTrue(report["contract"]["valid"])
            self.assertEqual(len(report["comparisons"]), 6)
            write_outputs(report, args.output_dir)
            self.assertTrue((args.output_dir / "parity_report.md").is_file())
            self.assertTrue((args.output_dir / "parity_report.json").is_file())
            self.assertTrue((args.output_dir / "parity_comparison.csv").is_file())

    def test_large_numerical_difference_warns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            official_path, csv_path = self.make_fixture(root, hard_offset=0.2)
            report = build_report(self.args(root, official_path, csv_path))
            self.assertEqual(report["verdict"], "WARN")
            failed = [
                item for item in report["comparisons"] if not item["within_tolerance"]
            ]
            self.assertEqual([item["official_key"] for item in failed], ["cape-hard-chamfer"])

    def test_incomplete_case_contract_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            official_path, csv_path = self.make_fixture(root)
            lines = csv_path.read_text(encoding="utf-8").splitlines()
            csv_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            report = build_report(self.args(root, official_path, csv_path))
            self.assertEqual(report["verdict"], "FAIL")
            self.assertFalse(report["contract"]["checks"]["case_count_450"])

    def test_official_loader_accepts_torch_scalar_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local test environment")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "torch-results.npy"
            np.save(path, {"cape-easy-chamfer": torch.tensor(0.8)}, allow_pickle=True)
            loaded = load_official(path)
            self.assertAlmostEqual(loaded["cape-easy-chamfer"], 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
