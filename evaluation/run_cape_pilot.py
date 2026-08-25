"""Run a bounded or complete, auditable ICON evaluation on CAPE.

The official benchmark assumes all 150 subjects and three rotations are
present when it aggregates results.  This entry point deliberately bypasses
that fixed-size epoch aggregation while reusing the official dataset,
checkpoint loader, ICON ``test_step``, reconstruction engine, and Evaluator.
It does not change the network or metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from uuid import uuid4


RUNNER_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_status(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def move_to_device(value: Any, device: Any) -> Any:
    """Move tensors in a collated batch without changing strings or metadata."""
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def write_csv(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ICON-filter on a bounded subset or the complete 150-subject "
            "x 3-rotation CAPE evaluation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/train/icon-filter.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./evaluation/outputs/cape-pilot"),
    )
    parser.add_argument(
        "--subject-indices",
        type=int,
        nargs="+",
        default=[0, 50],
        help="Zero-based positions in CAPE test.txt; defaults to easy[0], hard[0].",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        nargs="+",
        default=[0],
        help="CAPE view angles to process; valid values are 0, 120, and 240.",
    )
    parser.add_argument("--mcube-res", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse cases whose metrics, artifacts, parameters, and checkpoint "
            "hashes match this run. Failed or incomplete cases are recomputed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, checkpoints, subjects, and indices without inference.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mcube_res <= 0:
        raise ValueError("--mcube-res must be positive")
    invalid_rotations = sorted(set(args.rotations) - {0, 120, 240})
    if invalid_rotations:
        raise ValueError(
            "CAPE pilot rotations must be chosen from 0, 120, 240; got "
            f"{invalid_rotations}"
        )
    for subject_index in args.subject_indices:
        if subject_index < 0 or subject_index >= 150:
            raise ValueError(
                f"CAPE subject index must be in [0, 149], got {subject_index}"
            )


def case_key(case: Mapping[str, Any]) -> tuple[int, int]:
    return int(case["subject_index"]), int(case["rotation"])


def case_directory(output_dir: Path, case: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / str(case["group"])
        / str(case["subject"])
        / f"rotation-{int(case['rotation']):03d}"
    )


def artifact_paths(case_dir: Path) -> Dict[str, Path]:
    return {
        "prediction_mesh": case_dir / "mesh_src.obj",
        "ground_truth_mesh": case_dir / "mesh_tgt.obj",
        "intermediate_image": case_dir / "intermediate.png",
        "normal_comparison_image": case_dir / "normal_comparison.png",
    }


def load_resumable_record(
    output_dir: Path,
    case: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> Dict[str, Any] | None:
    """Return a validated successful record or None when recomputation is safer."""
    case_dir = case_directory(output_dir, case)
    metrics_path = case_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("status") != "ok":
        return None
    comparisons = {
        "subject_index": int(case["subject_index"]),
        "subject": str(case["subject"]),
        "rotation": int(case["rotation"]),
        **expected,
    }
    for key, value in comparisons.items():
        if record.get(key) != value:
            return None
    current_artifacts = artifact_paths(case_dir)
    if any(not path.is_file() or path.stat().st_size <= 0 for path in current_artifacts.values()):
        return None
    record.update({f"{name}_path": str(path) for name, path in current_artifacts.items()})
    return record


def audit_prediction_mesh(path: Path) -> Dict[str, Any]:
    """Record topology evidence without treating an imperfect prediction as missing."""
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    extents = [float(value) for value in mesh.extents]
    body_count = int(mesh.body_count)
    watertight = bool(mesh.is_watertight)
    winding_consistent = bool(mesh.is_winding_consistent)
    return {
        "mesh_vertex_count": int(len(mesh.vertices)),
        "mesh_face_count": int(len(mesh.faces)),
        "mesh_watertight": watertight,
        "mesh_winding_consistent": winding_consistent,
        "mesh_body_count": body_count,
        "mesh_extents": extents,
        "mesh_topology_ok": watertight and winding_consistent and body_count == 1,
    }


def ordered_records(
    selected_cases: Iterable[Mapping[str, Any]],
    records_by_key: Mapping[tuple[int, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        records_by_key[case_key(case)]
        for case in selected_cases
        if case_key(case) in records_by_key
    ]


def write_run_summary(
    output_dir: Path,
    selected_cases: List[Dict[str, Any]],
    records_by_key: Mapping[tuple[int, int], Dict[str, Any]],
    repo_root: Path,
    complete: bool,
) -> Dict[str, Any]:
    records = ordered_records(selected_cases, records_by_key)
    failures = sum(record.get("status") != "ok" for record in records)
    topology_warnings = sum(
        record.get("status") == "ok" and record.get("mesh_topology_ok") is False
        for record in records
    )
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "complete": complete,
        "selected_case_count": len(selected_cases),
        "case_count": len(records),
        "success_count": len(records) - failures,
        "failure_count": failures,
        "pending_count": len(selected_cases) - len(records),
        "topology_warning_count": topology_warnings,
        "records": records,
    }
    write_csv(output_dir / "pilot_summary.csv", records)
    with (output_dir / "pilot_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main() -> int:
    args = parse_args()
    validate_args(args)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"ICON config does not exist: {config_path}")

    # Heavy imports stay inside main so --help and static inspection remain cheap.
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate

    from apps.ICON import ICON
    from lib.common.config import get_cfg_defaults
    from lib.common.train_util import load_networks
    from lib.dataset.PIFuDataset import PIFuDataset

    if args.device != "cuda:0":
        raise ValueError(
            "The verified ICON environment and official config use cuda:0; "
            f"pilot received {args.device!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ICON CAPE pilot")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list(
        [
            "test_mode",
            True,
            "dataset.types",
            ["cape"],
            "dataset.scales",
            [100.0],
            "dataset.rotation_num",
            3,
            "mcube_res",
            args.mcube_res,
            "clean_mesh",
            True,
            "results_path",
            str(output_dir / "official-artifacts"),
        ]
    )
    cfg.freeze()

    checkpoint_path = (repo_root / cfg.resume_path).resolve()
    normal_path = (repo_root / cfg.normal_path).resolve()
    # PIFuDataset reads its dataset root from cfg.root (not cfg.dataset.root).
    test_list_path = (repo_root / cfg.root / "cape" / "test.txt").resolve()
    required_paths = {
        "checkpoint": checkpoint_path,
        "normal_checkpoint": normal_path,
        "test_list": test_list_path,
    }
    missing = {name: path for name, path in required_paths.items() if not path.is_file()}
    if missing:
        details = ", ".join(f"{name}={path}" for name, path in missing.items())
        raise FileNotFoundError(f"Required CAPE pilot assets are missing: {details}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    dataset = PIFuDataset(cfg, split="test")
    rotations = [int(value) for value in dataset.rotations]
    if len(dataset.subject_list) != 150 or rotations != [0, 120, 240]:
        raise RuntimeError(
            "Pilot requires the official CAPE layout: 150 subjects and rotations "
            f"[0, 120, 240]; got {len(dataset.subject_list)} and {rotations}."
        )

    selected_cases: List[Dict[str, Any]] = []
    for subject_index in args.subject_indices:
        group = "easy" if subject_index < 50 else "hard"
        subject_key = str(dataset.subject_list[subject_index])
        subject_name = subject_key.split("/", 1)[-1]
        for rotation in args.rotations:
            rotation_index = rotations.index(rotation)
            dataset_index = subject_index * len(rotations) + rotation_index
            selected_cases.append(
                {
                    "group": group,
                    "subject_index": subject_index,
                    "subject": subject_name,
                    "rotation": rotation,
                    "dataset_index": dataset_index,
                }
            )

    environment_record = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    }
    selection_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "resume_requested": args.resume,
        "git_commit": git_commit(repo_root),
        "git_status_short": git_status(repo_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "normal_checkpoint": str(normal_path),
        "mcube_res": args.mcube_res,
        "seed": args.seed,
        "device": args.device,
        "environment": environment_record,
        "cases": selected_cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selection.json").open("w", encoding="utf-8") as stream:
        json.dump(selection_record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(selection_record, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("DRY_RUN_OK: no model inference was executed")
        return 0

    checkpoint_hashes = {
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normal_checkpoint_sha256": sha256_file(normal_path),
    }
    resume_contract = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "method": cfg.name,
        "mcube_res": args.mcube_res,
        "seed": args.seed,
        "device": args.device,
        **checkpoint_hashes,
    }
    execution_session_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex[:8]
    )
    records_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    if args.resume:
        for case in selected_cases:
            cached = load_resumable_record(output_dir, case, resume_contract)
            if cached is not None:
                cached["resume_reused"] = True
                cached["resume_reused_at_utc"] = datetime.now(timezone.utc).isoformat()
                records_by_key[case_key(case)] = cached
        print(
            f"RESUME_SCAN reusable={len(records_by_key)} "
            f"recompute={len(selected_cases) - len(records_by_key)}"
        )

    cases_to_compute = [
        case for case in selected_cases if case_key(case) not in records_by_key
    ]
    if not cases_to_compute:
        summary = write_run_summary(
            output_dir, selected_cases, records_by_key, repo_root, complete=True
        )
        print("RESUME_COMPLETE: every selected case was already valid")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    device = torch.device(args.device)
    model_started = time.perf_counter()
    model = ICON(cfg)
    load_networks(
        cfg,
        model,
        mlp_path=str(checkpoint_path),
        normal_path=str(normal_path),
    )
    model.to(device)
    model.eval()
    model.netG.eval()
    model.netG.training = False
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_started

    session_has_success = False
    for case_number, case in enumerate(selected_cases, start=1):
        if case_key(case) in records_by_key:
            print(
                f"PILOT_CASE_SKIPPED {case_number}/{len(selected_cases)} "
                f"subject={case['subject']} rotation={case['rotation']}"
            )
            continue

        case_dir = case_directory(output_dir, case)
        case_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"PILOT_CASE {case_number}/{len(selected_cases)} "
            f"group={case['group']} subject={case['subject']} "
            f"rotation={case['rotation']} dataset_index={case['dataset_index']}"
        )

        record: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": cfg.name,
            "group": case["group"],
            "subject_index": case["subject_index"],
            "subject": case["subject"],
            "rotation": case["rotation"],
            "dataset_index": case["dataset_index"],
            "mcube_res": args.mcube_res,
            "seed": args.seed,
            "device": args.device,
            "torch_version": environment_record["torch_version"],
            "torch_cuda_version": environment_record["torch_cuda_version"],
            "gpu_name": environment_record["gpu_name"],
            "model_load_seconds": model_load_seconds,
            "execution_session_id": execution_session_id,
            "resume_reused": False,
            "warmup_affected": None,
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "status": "failed",
            "chamfer_cm": None,
            "p2s_cm": None,
            "normal_error": None,
            "runtime_seconds": None,
            "peak_memory_allocated_gib": None,
            "peak_memory_reserved_gib": None,
            **checkpoint_hashes,
        }

        try:
            sample = dataset[case["dataset_index"]]
            batch = default_collate([sample])
            batch = move_to_device(batch, device)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                metrics = model.test_step(batch, case["dataset_index"])
            torch.cuda.synchronize(device)
            runtime_seconds = time.perf_counter() - started

            model.evaluator.export_mesh(str(case_dir), "mesh")
            official_dir = Path(model.export_dir)
            for source_name, target_name in (
                (f"{case['rotation']}_inter.png", "intermediate.png"),
                (f"{case['rotation']}_nc.png", "normal_comparison.png"),
            ):
                source = official_dir / source_name
                if source.is_file():
                    shutil.copy2(str(source), str(case_dir / target_name))

            current_artifacts = artifact_paths(case_dir)
            missing_artifacts = {
                name: path
                for name, path in current_artifacts.items()
                if not path.is_file() or path.stat().st_size <= 0
            }
            if missing_artifacts:
                details = ", ".join(
                    f"{name}={path}" for name, path in missing_artifacts.items()
                )
                raise FileNotFoundError(
                    f"ICON metrics ran but required evidence is missing: {details}"
                )

            record.update(
                {
                    "status": "ok",
                    "warmup_affected": not session_has_success,
                    "chamfer_cm": as_float(metrics["chamfer"]),
                    "p2s_cm": as_float(metrics["p2s"]),
                    "normal_error": as_float(metrics["NC"]),
                    "runtime_seconds": runtime_seconds,
                    "peak_memory_allocated_gib": (
                        torch.cuda.max_memory_allocated(device) / (1024**3)
                    ),
                    "peak_memory_reserved_gib": (
                        torch.cuda.max_memory_reserved(device) / (1024**3)
                    ),
                    **{
                        f"{name}_path": str(path)
                        for name, path in current_artifacts.items()
                    },
                    **audit_prediction_mesh(current_artifacts["prediction_mesh"]),
                }
            )
            stale_traceback = case_dir / "traceback.txt"
            if stale_traceback.is_file():
                stale_traceback.unlink()
        except Exception as exc:  # Keep evidence for both pilot cases.
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)
            with (case_dir / "traceback.txt").open("w", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            print(f"PILOT_CASE_FAILED: {type(exc).__name__}: {exc}")

        if record["status"] == "ok":
            session_has_success = True
        records_by_key[case_key(case)] = record
        with (case_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        write_run_summary(
            output_dir, selected_cases, records_by_key, repo_root, complete=False
        )

    summary = write_run_summary(
        output_dir, selected_cases, records_by_key, repo_root, complete=True
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 and summary["pending_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
