"""Compare ICON's official CAPE aggregate with the auditable case-level runner.

This module does not run model inference.  It verifies that the case-level CSV
written by :mod:`evaluation.run_cape_pilot` reproduces the aggregation contract
used by ``apps.ICON.test_epoch_end`` and ``lib.common.train_util.accumulate``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


GROUP_COUNTS = {"easy": 150, "hard": 300}
ROTATIONS = (0, 120, 240)
METRICS = {
    "chamfer": "chamfer_cm",
    "p2s": "p2s_cm",
    "NC": "normal_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit numerical parity between ICON's official CAPE test_results.npy "
            "and evaluation.run_cape_pilot's 450-case CSV."
        )
    )
    parser.add_argument("--official-npy", type=Path, required=True)
    parser.add_argument("--custom-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/outputs/official-parity"),
    )
    parser.add_argument(
        "--surface-abs-tol-cm",
        type=float,
        default=0.03,
        help="Engineering parity tolerance for Chamfer and P2S, in centimetres.",
    )
    parser.add_argument(
        "--normal-abs-tol",
        type=float,
        default=0.003,
        help="Engineering parity tolerance for ICON's normal-error value.",
    )
    parser.add_argument(
        "--relative-tol",
        type=float,
        default=0.03,
        help="Relative tolerance used together with the metric-specific absolute tolerance.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any, label: str) -> float:
    """Convert numpy/torch/python scalar-like values into a finite float."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Official value {label!r} is not scalar: shape={array.shape}")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"Official value {label!r} is not finite: {result}")
    return result


@contextmanager
def cpu_portable_torch_pickle():
    """Map CUDA torch storages embedded in numpy object files onto the CPU.

    ICON's official ``test_results.npy`` may contain torch scalar tensors saved
    from a CUDA process.  NumPy delegates those objects back to ``torch.load``
    while unpickling, which otherwise fails on a CPU-only analysis machine.
    """

    try:
        import torch
    except ImportError:
        yield
        return

    original = torch.storage._load_from_bytes

    def load_from_bytes(payload: bytes):
        return torch.load(io.BytesIO(payload), map_location="cpu")

    torch.storage._load_from_bytes = load_from_bytes
    try:
        yield
    finally:
        torch.storage._load_from_bytes = original


def load_official(path: Path) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Official test_results.npy does not exist: {path}")
    with cpu_portable_torch_pickle():
        payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Official result must be the dictionary saved by ICON.test_epoch_end; "
            f"got {type(payload).__name__}."
        )
    return {str(key): scalar(value, str(key)) for key, value in payload.items()}


def parse_finite_float(value: Any, label: str) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV value {label!r} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"CSV value {label!r} is not finite: {number}")
    return number


def load_custom(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Custom pilot_summary.csv does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"Custom CSV contains no case rows: {path}")
    required = {"status", "group", "subject", "rotation", *METRICS.values()}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Custom CSV is missing required columns: {missing}")
    for index, row in enumerate(rows, start=2):
        row["rotation"] = int(str(row["rotation"]).strip())
        for column in METRICS.values():
            row[column] = parse_finite_float(row[column], f"row {index} {column}")
    return rows


def validate_contract(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures = [row for row in rows if str(row.get("status", "")).lower() != "ok"]
    group_counts = Counter(str(row.get("group")) for row in rows)
    rotation_counts = Counter(int(row["rotation"]) for row in rows)
    identities = [(str(row["subject"]), int(row["rotation"])) for row in rows]
    duplicate_count = len(identities) - len(set(identities))
    unique_subjects = len({str(row["subject"]) for row in rows})

    checks = {
        "case_count_450": len(rows) == 450,
        "success_count_450": len(failures) == 0 and len(rows) == 450,
        "unique_subject_count_150": unique_subjects == 150,
        "group_counts_match_official": dict(group_counts) == GROUP_COUNTS,
        "rotation_counts_match_official": dict(rotation_counts)
        == {rotation: 150 for rotation in ROTATIONS},
        "no_duplicate_subject_rotation": duplicate_count == 0,
    }
    return {
        "checks": checks,
        "valid": all(checks.values()),
        "case_count": len(rows),
        "failure_count": len(failures),
        "unique_subject_count": unique_subjects,
        "group_counts": dict(sorted(group_counts.items())),
        "rotation_counts": {str(k): v for k, v in sorted(rotation_counts.items())},
        "duplicate_subject_rotation_count": duplicate_count,
    }


def group_mean(
    rows: Iterable[Mapping[str, Any]], group: str, column: str
) -> float:
    values = [float(row[column]) for row in rows if str(row["group"]) == group]
    if not values:
        raise ValueError(f"No custom rows were found for group={group!r}")
    return math.fsum(values) / len(values)


def expected_official_keys() -> List[str]:
    return [
        f"cape-{group}-{metric}"
        for group in ("easy", "hard")
        for metric in METRICS
    ]


def compare(
    official: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    surface_abs_tol_cm: float,
    normal_abs_tol: float,
    relative_tol: float,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    expected = expected_official_keys()
    missing = [key for key in expected if key not in official]
    unexpected = sorted(key for key in official if key not in expected)
    comparisons: List[Dict[str, Any]] = []
    for group in ("easy", "hard"):
        for official_metric, csv_column in METRICS.items():
            key = f"cape-{group}-{official_metric}"
            custom_value = group_mean(rows, group, csv_column)
            if key not in official:
                comparisons.append(
                    {
                        "group": group,
                        "metric": official_metric,
                        "official_key": key,
                        "official_value": None,
                        "custom_value": custom_value,
                        "absolute_delta": None,
                        "relative_delta": None,
                        "tolerance": None,
                        "within_tolerance": False,
                    }
                )
                continue
            official_value = float(official[key])
            absolute_delta = abs(official_value - custom_value)
            scale = max(abs(official_value), abs(custom_value))
            relative_delta = absolute_delta / scale if scale else 0.0
            absolute_tolerance = (
                normal_abs_tol if official_metric == "NC" else surface_abs_tol_cm
            )
            tolerance = max(absolute_tolerance, relative_tolerance(scale, relative_tol))
            comparisons.append(
                {
                    "group": group,
                    "metric": official_metric,
                    "official_key": key,
                    "official_value": official_value,
                    "custom_value": custom_value,
                    "absolute_delta": absolute_delta,
                    "relative_delta": relative_delta,
                    "tolerance": tolerance,
                    "within_tolerance": absolute_delta <= tolerance,
                }
            )
    return comparisons, missing, unexpected


def relative_tolerance(scale: float, relative_tol: float) -> float:
    return scale * relative_tol


def overall_status(
    contract: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    missing_keys: Sequence[str],
) -> str:
    if not contract["valid"] or missing_keys:
        return "FAIL"
    if all(bool(item["within_tolerance"]) for item in comparisons):
        return "PASS"
    return "WARN"


def format_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8f}"


def markdown_report(report: Mapping[str, Any]) -> str:
    contract = report["contract"]
    lines = [
        "# ICON official-vs-custom CAPE parity audit",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        "This is a measurement-pipeline calibration. It does not rerun ICON and it "
        "does not reproduce a paper table by itself.",
        "",
        "## Input contract",
        "",
        f"- Cases: {contract['case_count']} (expected 450)",
        f"- Successful cases: {contract['case_count'] - contract['failure_count']}",
        f"- Unique subjects: {contract['unique_subject_count']} (expected 150)",
        f"- Group counts: `{contract['group_counts']}`",
        f"- Rotation counts: `{contract['rotation_counts']}`",
        f"- Duplicate subject/rotation pairs: {contract['duplicate_subject_rotation_count']}",
        "",
        "## Numerical parity",
        "",
        "| Group | Metric | Official | Custom case mean | Absolute delta | Allowed delta | Result |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in report["comparisons"]:
        result = "PASS" if item["within_tolerance"] else "WARN"
        lines.append(
            f"| {item['group']} | {item['metric']} | "
            f"{format_number(item['official_value'])} | "
            f"{format_number(item['custom_value'])} | "
            f"{format_number(item['absolute_delta'])} | "
            f"{format_number(item['tolerance'])} | {result} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS`: the two routes agree within the declared engineering tolerance.",
            "- `WARN`: both result files are structurally valid, but at least one "
            "aggregate differs beyond tolerance; inspect random sampling, ordering, "
            "configuration hashes, and checkpoints before comparing methods.",
            "- `FAIL`: the official keys or the complete 150 x 3 CAPE contract are missing.",
            "",
            "The tolerances are a pipeline-debugging rule, not a confidence interval and "
            "not a new paper metric. Exact deltas remain in the JSON/CSV evidence.",
        ]
    )
    if report["missing_official_keys"]:
        lines.extend(
            ["", f"Missing official keys: `{report['missing_official_keys']}`"]
        )
    if report["unexpected_official_keys"]:
        lines.extend(
            ["", f"Unexpected official keys: `{report['unexpected_official_keys']}`"]
        )
    return "\n".join(lines) + "\n"


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "parity_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    with (output_dir / "parity_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "group",
            "metric",
            "official_key",
            "official_value",
            "custom_value",
            "absolute_delta",
            "relative_delta",
            "tolerance",
            "within_tolerance",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["comparisons"])


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    official_path = args.official_npy.resolve()
    custom_path = args.custom_csv.resolve()
    official = load_official(official_path)
    rows = load_custom(custom_path)
    contract = validate_contract(rows)
    comparisons, missing, unexpected = compare(
        official,
        rows,
        surface_abs_tol_cm=args.surface_abs_tol_cm,
        normal_abs_tol=args.normal_abs_tol,
        relative_tol=args.relative_tol,
    )
    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "",
        "inputs": {
            "official_npy": str(official_path),
            "official_npy_sha256": sha256_file(official_path),
            "custom_csv": str(custom_path),
            "custom_csv_sha256": sha256_file(custom_path),
        },
        "aggregation_contract": (
            "arithmetic mean over 50 easy or 100 hard subjects, each at rotations "
            "0/120/240, matching lib.common.train_util.accumulate"
        ),
        "tolerances": {
            "surface_abs_tol_cm": args.surface_abs_tol_cm,
            "normal_abs_tol": args.normal_abs_tol,
            "relative_tol": args.relative_tol,
            "rule": "max(metric_absolute_tolerance, relative_tol * max(abs(values)))",
        },
        "contract": contract,
        "official_values": official,
        "missing_official_keys": missing,
        "unexpected_official_keys": unexpected,
        "comparisons": comparisons,
    }
    report["verdict"] = overall_status(contract, comparisons, missing)
    return report


def main() -> int:
    args = parse_args()
    report = build_report(args)
    write_outputs(report, args.output_dir.resolve())
    print(markdown_report(report))
    print(f"Wrote parity evidence to {args.output_dir.resolve()}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
