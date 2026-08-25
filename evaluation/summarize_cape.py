"""Summarize an auditable CAPE pilot or benchmark CSV.

The runner writes one row per subject-view case.  This module turns those
case-level records into descriptive statistics without changing the metric
definitions or silently dropping failures.  It intentionally uses only the
Python standard library so it can run in the verified ICON environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


QUALITY_METRICS = ("chamfer_cm", "p2s_cm", "normal_error")
SYSTEM_METRICS = (
    "runtime_seconds",
    "peak_memory_allocated_gib",
    "peak_memory_reserved_gib",
)
ALL_METRICS = QUALITY_METRICS + SYSTEM_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CAPE case-level CSV results into JSON, CSV, and Markdown."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/outputs/cape-smoke-30/pilot_summary.csv"),
        help="Case-level CSV written by evaluation.run_cape_pilot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to an analysis/ directory beside the input CSV.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> Dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "p90": percentile(numbers, 0.90),
        "min": min(numbers),
        "max": max(numbers),
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CAPE summary CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"CAPE summary CSV contains no case rows: {path}")
    required = {"status", "group", "subject", "rotation", *QUALITY_METRICS}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"CAPE summary CSV is missing columns: {missing}")
    for row in rows:
        for metric in ALL_METRICS:
            row[metric] = optional_float(row.get(metric))
        for key in ("subject_index", "rotation", "dataset_index", "mcube_res", "seed"):
            value = str(row.get(key, "")).strip()
            row[key] = int(value) if value else None
    return rows


def metric_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        metric: describe(
            row[metric]
            for row in rows
            if row.get(metric) is not None
        )
        for metric in ALL_METRICS
    }


def group_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        label = ", ".join(f"{key}={row.get(key)}" for key in keys)
        groups.setdefault(label, []).append(row)
    return groups


def case_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "group": row.get("group"),
        "subject_index": row.get("subject_index"),
        "subject": row.get("subject"),
        "rotation": row.get("rotation"),
        "dataset_index": row.get("dataset_index"),
    }


def extremes(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> Dict[str, Any] | None:
    available = [row for row in rows if row.get(metric) is not None]
    if not available:
        return None
    best = min(available, key=lambda row: float(row[metric]))
    worst = max(available, key=lambda row: float(row[metric]))
    return {
        "direction": "lower_is_better",
        "best": {**case_identity(best), "value": best[metric]},
        "worst": {**case_identity(worst), "value": worst[metric]},
    }


def build_summary(rows: Sequence[Mapping[str, Any]], input_path: Path) -> Dict[str, Any]:
    successful = [row for row in rows if str(row.get("status")).lower() == "ok"]
    failed = [row for row in rows if str(row.get("status")).lower() != "ok"]
    steady_state = successful[1:] if len(successful) > 1 else []

    by_group = {
        label: metric_block(items)
        for label, items in group_rows(successful, ("group",)).items()
    }
    by_rotation = {
        label: metric_block(items)
        for label, items in group_rows(successful, ("rotation",)).items()
    }
    by_group_rotation = {
        label: metric_block(items)
        for label, items in group_rows(successful, ("group", "rotation")).items()
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "case_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(failed),
        "success_rate": len(successful) / len(rows),
        "unique_subject_count": len({row.get("subject") for row in rows}),
        "rotations": sorted(
            {row.get("rotation") for row in rows if row.get("rotation") is not None}
        ),
        "metric_contract": {
            "chamfer_cm": "lower_is_better; symmetric point-to-surface distance in cm",
            "p2s_cm": "lower_is_better; ground-truth samples to prediction surface in cm",
            "normal_error": "lower_is_better; official four-view rendered-normal squared error",
        },
        "overall": metric_block(successful),
        "by_group": by_group,
        "by_rotation": by_rotation,
        "by_group_rotation": by_group_rotation,
        "steady_state_system_metrics": {
            "note": (
                "The first successful case is excluded because CUDA lazy initialization "
                "and allocator warm-up can inflate runtime and peak memory."
            ),
            "excluded_case": case_identity(successful[0]) if successful else None,
            "statistics": {
                metric: describe(
                    row[metric]
                    for row in steady_state
                    if row.get(metric) is not None
                )
                for metric in SYSTEM_METRICS
            },
        },
        "quality_extremes": {
            metric: extremes(successful, metric) for metric in QUALITY_METRICS
        },
        "failures": [
            {
                **case_identity(row),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in failed
        ],
    }


def flatten_statistics(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    dimensions = {
        "overall": {"all": summary["overall"]},
        "group": summary["by_group"],
        "rotation": summary["by_rotation"],
        "group_rotation": summary["by_group_rotation"],
    }
    for dimension, groups in dimensions.items():
        for group, metrics in groups.items():
            for metric, stats in metrics.items():
                output.append(
                    {
                        "dimension": dimension,
                        "group": group,
                        "metric": metric,
                        **stats,
                    }
                )
    return output


def format_number(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# CAPE benchmark summary",
        "",
        "## Run validity",
        "",
        f"- Cases: **{summary['case_count']}**",
        f"- Unique subjects: **{summary['unique_subject_count']}**",
        f"- Rotations: **{', '.join(map(str, summary['rotations']))}**",
        f"- Successes: **{summary['success_count']}**",
        f"- Failures: **{summary['failure_count']}**",
        f"- Success rate: **{summary['success_rate'] * 100:.2f}%**",
        "",
        "## Overall quality metrics",
        "",
        "| Metric | Count | Mean | Median | Std | P90 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in QUALITY_METRICS:
        stats = summary["overall"][metric]
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    format_number(stats["count"], 0),
                    format_number(stats["mean"]),
                    format_number(stats["median"]),
                    format_number(stats["std"]),
                    format_number(stats["p90"]),
                    format_number(stats["min"]),
                    format_number(stats["max"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Quality by group", ""])
    for label, metrics in summary["by_group"].items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Metric | Count | Mean | Median | P90 |")
        lines.append("|---|---:|---:|---:|---:|")
        for metric in QUALITY_METRICS:
            stats = metrics[metric]
            lines.append(
                f"| {metric} | {stats['count']} | {format_number(stats['mean'])} | "
                f"{format_number(stats['median'])} | {format_number(stats['p90'])} |"
            )
        lines.append("")

    lines.extend(["## Quality by rotation", ""])
    lines.append("| Rotation | Chamfer mean (cm) | P2S mean (cm) | Normal-error mean |")
    lines.append("|---|---:|---:|---:|")
    for label, metrics in summary["by_rotation"].items():
        lines.append(
            f"| {label} | {format_number(metrics['chamfer_cm']['mean'])} | "
            f"{format_number(metrics['p2s_cm']['mean'])} | "
            f"{format_number(metrics['normal_error']['mean'])} |"
        )

    lines.extend(["", "## Best and worst observed cases", ""])
    lines.append("| Metric | Best case | Best | Worst case | Worst |")
    lines.append("|---|---|---:|---|---:|")
    for metric, item in summary["quality_extremes"].items():
        if item is None:
            continue
        best = item["best"]
        worst = item["worst"]
        best_name = f"{best['subject']} @ {best['rotation']}"
        worst_name = f"{worst['subject']} @ {worst['rotation']}"
        lines.append(
            f"| {metric} | {best_name} | {format_number(best['value'])} | "
            f"{worst_name} | {format_number(worst['value'])} |"
        )

    steady = summary["steady_state_system_metrics"]
    lines.extend(
        [
            "",
            "## System measurements after warm-up",
            "",
            steady["note"],
            "",
            "| Metric | Count | Mean | Median | P90 | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in SYSTEM_METRICS:
        stats = steady["statistics"][metric]
        lines.append(
            f"| {metric} | {stats['count']} | {format_number(stats['mean'])} | "
            f"{format_number(stats['median'])} | {format_number(stats['p90'])} | "
            f"{format_number(stats['max'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These are descriptive results for the cases listed in the input CSV. "
            "They are not a paper-table reproduction unless the complete official "
            "protocol, all 150 CAPE subjects, all three rotations, and every baseline "
            "use the same evaluation contract.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    flat_rows = flatten_statistics(summary)
    with (output_dir / "benchmark_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    with (output_dir / "benchmark_report.md").open("w", encoding="utf-8") as stream:
        stream.write(markdown_report(summary))


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_path.parent / "analysis"
    )
    rows = load_rows(input_path)
    summary = build_summary(rows, input_path)
    write_outputs(summary, output_dir)
    print(
        "SUMMARY_OK "
        f"cases={summary['case_count']} successes={summary['success_count']} "
        f"failures={summary['failure_count']} output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
