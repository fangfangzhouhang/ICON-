"""Paired CAPE experiment for prepared-SMPL versus image-estimated PIXIE prior.

Route A reuses the auditable CAPE outputs produced by ``run_cape_pilot``.
Route B feeds the exact same CAPE render to ``apps.infer`` and therefore lets
PIXIE estimate SMPL-X from the image.  Both resulting meshes are evaluated
against the same ground-truth mesh with the same metric implementation and
sampling seed.

This is an end-to-end route comparison, not a perfectly isolated causal
ablation: the two routes also differ in body-model family and preprocessing.
No ICP, best-fit rescaling, or centroid alignment is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from evaluation.coordinate_adapter import apply_homogeneous_transform


RUNNER_SCHEMA_VERSION = 1


def parse_int_list(values: Sequence[int]) -> List[int]:
    return list(dict.fromkeys(int(value) for value in values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "prepare", "infer", "evaluate"),
        default="all",
        help="Run the complete route or one resumable stage.",
    )
    parser.add_argument(
        "--prepared-csv",
        type=Path,
        default=Path("evaluation/outputs/cape-full-450/pilot_summary.csv"),
        help="Case-level output from evaluation.run_cape_pilot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/outputs/prior-source-ab"),
    )
    parser.add_argument("--subject-indices", nargs="+", type=int, default=[0, 50])
    parser.add_argument("--rotations", nargs="+", type=int, default=[0])
    parser.add_argument("--config", type=Path, default=Path("configs/icon-filter.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--loop-smpl", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument(
        "--max-anisotropy",
        type=float,
        default=1.001,
        help="Reject traces whose x/y crop scales disagree beyond this ratio.",
    )
    return parser.parse_args()


def resolve_from(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared-route CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    required = {
        "status",
        "group",
        "subject_index",
        "subject",
        "rotation",
        "prediction_mesh_path",
        "ground_truth_mesh_path",
    }
    if not rows:
        raise ValueError(f"Prepared-route CSV is empty: {path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Prepared-route CSV is missing columns: {missing}")
    return rows


def select_cases(
    rows: Sequence[Mapping[str, str]],
    subject_indices: Sequence[int],
    rotations: Sequence[int],
) -> List[Dict[str, Any]]:
    wanted_subjects = set(parse_int_list(subject_indices))
    wanted_rotations = set(parse_int_list(rotations))
    invalid = sorted(wanted_rotations - {0, 120, 240})
    if invalid:
        raise ValueError(f"CAPE rotations must be 0, 120, or 240; got {invalid}")

    cases: List[Dict[str, Any]] = []
    for row in rows:
        if str(row["status"]).strip().lower() != "ok":
            continue
        subject_index = int(row["subject_index"])
        rotation = int(row["rotation"])
        if subject_index not in wanted_subjects or rotation not in wanted_rotations:
            continue
        case = dict(row)
        case["subject_index"] = subject_index
        case["rotation"] = rotation
        case["sample_name"] = f"ab-s{subject_index:03d}-r{rotation:03d}"
        cases.append(case)

    expected = len(wanted_subjects) * len(wanted_rotations)
    if len(cases) != expected:
        found = [(case["subject_index"], case["rotation"]) for case in cases]
        raise ValueError(f"Expected {expected} selected cases, found {len(cases)}: {found}")
    return sorted(cases, key=lambda item: (item["subject_index"], item["rotation"]))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    fields: List[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def prepare_inputs(repo_root: Path, output_dir: Path, cases: List[Dict[str, Any]]) -> Path:
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    selected_names = {str(case["sample_name"]) for case in cases}
    for stale_input in input_dir.glob("ab-s*-r*.png"):
        if stale_input.stem not in selected_names:
            stale_input.unlink()
    manifest: List[Dict[str, Any]] = []
    for case in cases:
        source = (
            repo_root
            / "data"
            / "cape_3views"
            / str(case["subject"])
            / "render"
            / f"{int(case['rotation']):03d}.png"
        )
        if not source.is_file():
            raise FileNotFoundError(f"CAPE render is missing: {source}")
        target = input_dir / f"{case['sample_name']}.png"
        shutil.copy2(str(source), str(target))
        manifest.append(
            {
                "sample_name": case["sample_name"],
                "group": case["group"],
                "subject_index": case["subject_index"],
                "subject": case["subject"],
                "rotation": case["rotation"],
                "source_image_path": str(source.resolve()),
                "prepared_prediction_mesh_path": case["prediction_mesh_path"],
                "ground_truth_mesh_path": case["ground_truth_mesh_path"],
            }
        )
    write_csv(output_dir / "selection.csv", manifest)
    with (output_dir / "selection.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return input_dir


def run_inference(
    repo_root: Path,
    output_dir: Path,
    input_dir: Path,
    config_path: Path,
    gpu_device: int,
    loop_smpl: int,
) -> None:
    inference_dir = output_dir / "pixie-inference"
    sample_names = {path.stem for path in input_dir.glob("ab-s*-r*.png")}
    if not sample_names:
        raise FileNotFoundError(f"No prepared A/B input images exist below {input_dir}")

    # A failed rerun must never be allowed to inherit scientific evidence from
    # an older successful run with the same deterministic sample name.
    for sample_name in sample_names:
        for suffix in ("_recon.obj", "_trace.json"):
            for stale_output in inference_dir.rglob(f"{sample_name}{suffix}"):
                stale_output.unlink()

    command = [
        sys.executable,
        "-m",
        "apps.infer",
        "-cfg",
        str(config_path),
        "-gpu",
        str(gpu_device),
        "-in_dir",
        str(input_dir),
        "-out_dir",
        str(inference_dir),
        "-loop_smpl",
        str(loop_smpl),
        "-hps_type",
        "pixie",
        "--stop-after-recon",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pixie-inference.log"
    print(f"Running PIXIE route for {len(sample_names)} case(s). Live log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "PIXIE-route inference failed with exit code "
            f"{completed.returncode}; inspect {log_path}"
        )
    print(f"PIXIE route finished successfully. Evidence: {inference_dir}", flush=True)


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern!r} below {root}, found {matches}")
    return matches[0]


def audit_mesh(path: Path) -> Dict[str, Any]:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "body_count": int(mesh.body_count),
        "extents": [float(value) for value in mesh.extents],
    }


def map_pixie_mesh(trace_path: Path, target_path: Path, max_anisotropy: float) -> Dict[str, Any]:
    import numpy as np
    import trimesh

    with trace_path.open("r", encoding="utf-8") as stream:
        trace = json.load(stream)
    transform = trace.get("crop_ndc_to_image_ndc", {})
    ratio = float(transform.get("anisotropy_ratio", float("inf")))
    if ratio > max_anisotropy:
        raise ValueError(
            f"Coordinate trace anisotropy {ratio:.8f} exceeds {max_anisotropy:.8f}"
        )
    source_path = Path(trace["mesh_path"])
    if not source_path.is_file():
        source_path = trace_path.with_name(trace_path.name.replace("_trace.json", "_recon.obj"))
    mesh = trimesh.load(str(source_path), force="mesh", process=False)
    vertices = apply_homogeneous_transform(mesh.vertices, np.asarray(transform["matrix"]))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    mapped = trimesh.Trimesh(vertices, mesh.faces, process=False, maintains_order=True)
    mapped.export(str(target_path))
    return {"trace": trace, "mesh": audit_mesh(target_path)}


def evaluate_cases(
    output_dir: Path,
    cases: List[Dict[str, Any]],
    device: str,
    num_samples: int,
    seed: int,
    max_anisotropy: float,
) -> List[Dict[str, Any]]:
    from evaluation.metrics.official_icon import evaluate_aligned_meshes

    inference_root = output_dir / "pixie-inference"
    records: List[Dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        sample_name = str(case["sample_name"])
        case_dir = output_dir / "cases" / sample_name
        case_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = Path(str(case["prediction_mesh_path"])).resolve()
        gt_path = Path(str(case["ground_truth_mesh_path"])).resolve()
        pixie_mesh_path = case_dir / "pixie_image_prior.obj"
        case_seed = seed + int(case["subject_index"]) * 3 + {
            0: 0,
            120: 1,
            240: 2,
        }[int(case["rotation"])]
        record: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "group": case["group"],
            "subject_index": case["subject_index"],
            "subject": case["subject"],
            "rotation": case["rotation"],
            "sample_name": sample_name,
            "metric_seed": case_seed,
            "num_samples": num_samples,
            "prepared_mesh_path": str(prepared_path),
            "pixie_mesh_path": str(pixie_mesh_path.resolve()),
            "ground_truth_mesh_path": str(gt_path),
        }
        for metric in ("chamfer_cm", "p2s_cm", "normal_error"):
            record[f"prepared_{metric}"] = ""
            record[f"pixie_{metric}"] = ""
            record[f"delta_{metric}"] = ""

        try:
            trace_path = find_one(inference_root, f"{sample_name}_trace.json")
            record["trace_path"] = str(trace_path.resolve())
            mapping = map_pixie_mesh(trace_path, pixie_mesh_path, max_anisotropy)
            record["coordinate_anisotropy_ratio"] = mapping["trace"][
                "crop_ndc_to_image_ndc"
            ]["anisotropy_ratio"]

            for label, path in (
                ("prepared prediction", prepared_path),
                ("ground truth", gt_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"{label} is missing: {path}")

            prepared_metrics = evaluate_aligned_meshes(
                prepared_path,
                gt_path,
                case_dir / "prepared_normal.png",
                device=device,
                num_samples=num_samples,
                seed=case_seed,
            )
            pixie_metrics = evaluate_aligned_meshes(
                pixie_mesh_path,
                gt_path,
                case_dir / "pixie_normal.png",
                device=device,
                num_samples=num_samples,
                seed=case_seed,
            )
            for metric in ("chamfer_cm", "p2s_cm", "normal_error"):
                prepared_value = float(prepared_metrics[metric])
                pixie_value = float(pixie_metrics[metric])
                record[f"prepared_{metric}"] = prepared_value
                record[f"pixie_{metric}"] = pixie_value
                record[f"delta_{metric}"] = pixie_value - prepared_value
            for key, value in mapping["mesh"].items():
                record[f"pixie_mesh_{key}"] = value
            record["status"] = "ok"
        except Exception as error:  # Preserve failures in the scientific denominator.
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            record["error_message"] = str(error)
        record["elapsed_seconds"] = time.perf_counter() - started

        with (case_dir / "paired_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        records.append(record)

    write_csv(output_dir / "paired_summary.csv", records)
    summary = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "scientific_question": "prepared CAPE SMPL prior versus image-estimated PIXIE SMPL-X route",
        "causal_isolation": False,
        "hidden_alignment": False,
        "selected_case_count": len(cases),
        "success_count": sum(record["status"] == "ok" for record in records),
        "failure_count": sum(record["status"] != "ok" for record in records),
        "records": records,
    }
    with (output_dir / "paired_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return records


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    prepared_csv = resolve_from(repo_root, args.prepared_csv)
    output_dir = resolve_from(repo_root, args.output_dir)
    config_path = resolve_from(repo_root, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"ICON inference config does not exist: {config_path}")
    if args.num_samples <= 0 or args.loop_smpl < 0:
        raise ValueError("--num-samples must be positive and --loop-smpl must be non-negative")

    cases = select_cases(load_csv(prepared_csv), args.subject_indices, args.rotations)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "inputs"
    if args.mode in ("all", "prepare"):
        input_dir = prepare_inputs(repo_root, output_dir, cases)
    if args.mode in ("all", "infer"):
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Prepared input directory does not exist: {input_dir}")
        run_inference(repo_root, output_dir, input_dir, config_path, args.gpu_device, args.loop_smpl)
    if args.mode in ("all", "evaluate"):
        records = evaluate_cases(
            output_dir,
            cases,
            args.device,
            args.num_samples,
            args.seed,
            args.max_anisotropy,
        )
        failure_count = sum(record["status"] != "ok" for record in records)
        print(
            json.dumps(
                {
                    "status": "ok" if failure_count == 0 else "failed",
                    "case_count": len(records),
                    "failure_count": failure_count,
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        if failure_count:
            return 2
    else:
        print(json.dumps({"status": "ok", "stage": args.mode, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
