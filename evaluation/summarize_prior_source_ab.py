"""Summarize the paired prepared-prior versus PIXIE-prior CAPE experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


METRICS = ("chamfer_cm", "p2s_cm", "normal_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Paired result CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"Paired result CSV is empty: {path}")
    required = {"status", "group", "subject", "rotation"}
    for metric in METRICS:
        required.update({f"prepared_{metric}", f"pixie_{metric}", f"delta_{metric}"})
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Paired result CSV is missing columns: {missing}")
    for row in rows:
        row["rotation"] = int(row["rotation"])
        if row["status"] == "ok":
            for metric in METRICS:
                for prefix in ("prepared_", "pixie_", "delta_"):
                    value = float(row[f"{prefix}{metric}"])
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite value in {prefix}{metric}: {value}")
                    row[f"{prefix}{metric}"] = value
    return rows


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of no values")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clustered_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    """Bootstrap subjects, keeping repeated rotations together."""

    by_subject: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(float(row[value_key]))
    subjects = sorted(by_subject)
    if len(subjects) == 1 or samples <= 0:
        mean = statistics.fmean(value for values in by_subject.values() for value in values)
        return mean, mean
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(samples):
        drawn = [rng.choice(subjects) for _ in subjects]
        values = [value for subject in drawn for value in by_subject[subject]]
        estimates.append(statistics.fmean(values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize_slice(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    success = [row for row in rows if row.get("status") == "ok"]
    result: Dict[str, Any] = {
        "slice": label,
        "selected_count": len(rows),
        "success_count": len(success),
        "failure_count": len(rows) - len(success),
        "unique_subject_count": len({str(row["subject"]) for row in success}),
        "metrics": {},
    }
    if not success:
        return result
    for index, metric in enumerate(METRICS):
        prepared = [float(row[f"prepared_{metric}"]) for row in success]
        pixie = [float(row[f"pixie_{metric}"]) for row in success]
        delta = [float(row[f"delta_{metric}"]) for row in success]
        ci_low, ci_high = clustered_bootstrap_ci(
            success,
            f"delta_{metric}",
            samples=bootstrap_samples,
            seed=seed + index,
        )
        result["metrics"][metric] = {
            "prepared_mean": statistics.fmean(prepared),
            "pixie_mean": statistics.fmean(pixie),
            "delta_mean_pixie_minus_prepared": statistics.fmean(delta),
            "delta_median": statistics.median(delta),
            "delta_p90": percentile(delta, 0.90),
            "delta_bootstrap_95_ci": [ci_low, ci_high],
            "pixie_worse_fraction": sum(value > 0 for value in delta) / len(delta),
        }
    return result


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    slices: List[Tuple[str, List[Mapping[str, Any]]]] = [("overall", list(rows))]
    for group in sorted({str(row["group"]) for row in rows}):
        slices.append((f"group={group}", [row for row in rows if row["group"] == group]))
    for rotation in sorted({int(row["rotation"]) for row in rows}):
        slices.append(
            (f"rotation={rotation}", [row for row in rows if int(row["rotation"]) == rotation])
        )
    return {
        "schema_version": 1,
        "delta_definition": "pixie_image_prior_minus_prepared_cape_prior; lower metrics are better",
        "bootstrap_unit": "subject (all selected rotations stay together)",
        "bootstrap_samples": bootstrap_samples,
        "scientific_boundary": (
            "End-to-end route comparison, not a pure one-variable ablation; body-model family, "
            "preprocessing, and feedback path also differ."
        ),
        "slices": [
            summarize_slice(
                label,
                members,
                bootstrap_samples=bootstrap_samples,
                seed=seed + slice_index * 100,
            )
            for slice_index, (label, members) in enumerate(slices)
        ],
    }


def markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Prior-source A/B report",
        "",
        "Delta is `PIXIE image-prior route - prepared CAPE-prior route`; positive means PIXIE is worse.",
        "",
        "This is an end-to-end route comparison, not a pure causal ablation.",
        "",
    ]
    for section in summary["slices"]:
        lines.extend(
            [
                f"## {section['slice']}",
                "",
                (
                    f"Selected={section['selected_count']}, success={section['success_count']}, "
                    f"failure={section['failure_count']}, subjects={section['unique_subject_count']}."
                ),
                "",
                "| Metric | Prepared mean | PIXIE mean | Mean delta | 95% cluster-bootstrap CI | PIXIE worse |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric, values in section["metrics"].items():
            ci = values["delta_bootstrap_95_ci"]
            lines.append(
                f"| {metric} | {values['prepared_mean']:.6f} | {values['pixie_mean']:.6f} | "
                f"{values['delta_mean_pixie_minus_prepared']:.6f} | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] | {values['pixie_worse_fraction']:.1%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation rules",
            "",
            "- Do not treat a two-case gate as a conclusion; it only validates coordinates and artifacts.",
            "- For repeated rotations, uncertainty is bootstrapped by subject rather than by image.",
            "- A confidence interval crossing zero means this sample does not establish a stable direction.",
            "- Failures remain in the denominator and must be reported separately.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    rows = load_rows(args.input_csv)
    summary = build_summary(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prior_source_ab_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    (args.output_dir / "prior_source_ab_report.md").write_text(
        markdown(summary), encoding="utf-8"
    )
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
