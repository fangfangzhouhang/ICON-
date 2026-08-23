"""CLI for auditable evaluation of one already-aligned mesh pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from evaluation.metrics import evaluate_aligned_meshes


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


def append_csv(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prediction and ground-truth meshes that are already in "
            "the same ICON coordinate frame. No alignment is performed."
        )
    )
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="Debug distance metrics only; output is not a complete benchmark row.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pred_path = args.pred.resolve()
    gt_path = args.gt.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    metrics = evaluate_aligned_meshes(
        pred_path,
        gt_path,
        output_dir / "normal_comparison.png",
        device=args.device,
        num_samples=args.num_samples,
        seed=args.seed,
        skip_normal=args.skip_normal,
    )
    runtime = time.perf_counter() - started

    record: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": args.sample_id,
        "method": args.method,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "pred_sha256": sha256_file(pred_path),
        "gt_sha256": sha256_file(gt_path),
        "git_commit": git_commit(repo_root),
        "num_samples": args.num_samples,
        "random_seed": args.seed,
        "device": args.device,
        "chamfer_cm": metrics["chamfer_cm"],
        "p2s_cm": metrics["p2s_cm"],
        "normal_error": metrics["normal_error"],
        "runtime_seconds": runtime,
        "complete_benchmark_row": not args.skip_normal,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    append_csv(output_dir / "metrics.csv", record)

    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
