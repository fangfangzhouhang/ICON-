import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_cape_pilot import (
    RUNNER_SCHEMA_VERSION,
    artifact_paths,
    case_directory,
    load_resumable_record,
    write_run_summary,
)


class CapeResumeTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "group": "easy",
            "subject_index": 0,
            "subject": "subject-a",
            "rotation": 0,
            "dataset_index": 0,
        }
        self.contract = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "method": "icon-filter",
            "mcube_res": 256,
            "seed": 1993,
            "device": "cuda:0",
            "config_sha256": "config",
            "checkpoint_sha256": "checkpoint",
            "normal_checkpoint_sha256": "normal",
        }

    def create_cached_case(self, root: Path):
        directory = case_directory(root, self.case)
        directory.mkdir(parents=True)
        for path in artifact_paths(directory).values():
            path.write_bytes(b"non-empty")
        record = {
            "status": "ok",
            "subject_index": 0,
            "subject": "subject-a",
            "rotation": 0,
            **self.contract,
        }
        (directory / "metrics.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        return directory

    def test_valid_successful_case_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_cached_case(root)
            record = load_resumable_record(root, self.case, self.contract)
            self.assertIsNotNone(record)
            self.assertTrue(record["prediction_mesh_path"].endswith("mesh_src.obj"))

    def test_changed_contract_forces_recomputation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_cached_case(root)
            changed = {**self.contract, "mcube_res": 512}
            self.assertIsNone(load_resumable_record(root, self.case, changed))

    def test_missing_artifact_forces_recomputation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self.create_cached_case(root)
            (directory / "mesh_src.obj").unlink()
            self.assertIsNone(load_resumable_record(root, self.case, self.contract))

    def test_progress_summary_keeps_pending_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = {(0, 0): {"status": "ok", "subject_index": 0, "rotation": 0}}
            second = {**self.case, "subject_index": 1, "subject": "subject-b"}
            summary = write_run_summary(
                root, [self.case, second], records, Path.cwd(), complete=False
            )
            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["pending_count"], 1)
            self.assertFalse(summary["complete"])


if __name__ == "__main__":
    unittest.main()
