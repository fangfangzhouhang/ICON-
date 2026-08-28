"""Compare multiple validated methods on exactly matched cases."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from evaluation.schema import (
    QUALITY_METRICS,
    duplicate_case_keys,
    load_csv,
    normalize_record,
    validate_record,
)


def percentile(values: Sequence[float], fraction: float) -> Any:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
    }


def parse_submission(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use METHOD=/path/to/unified.csv")
    method, path = value.split("=", 1)
    if not method.strip() or not path.strip():
        raise argparse.ArgumentTypeError("use METHOD=/path/to/unified.csv")
    return method.strip(), Path(path)


def load_submissions(specs: Sequence[Tuple[str, Path]]) -> Dict[str, List[Dict[str, Any]]]:
    submissions: Dict[str, List[Dict[str, Any]]] = {}
    for method, path in specs:
        if method in submissions:
            raise ValueError("duplicate method label: %s" % method)
        rows = load_csv(path)
        for row in rows:
            row["method"] = method
            row["method_variant"] = row.get("method_variant") or method
            errors = validate_record(row)
            if errors:
                raise ValueError("%s:%s: %s" % (path, row.get("sample_id"), "; ".join(errors)))
        duplicates = duplicate_case_keys(rows)
        if duplicates:
            raise ValueError("%s: duplicate method/sample keys: %s" % (path, duplicates))
        submissions[method] = rows
    return submissions


def build_comparison(submissions: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    normalized_submissions: Dict[str, List[Dict[str, Any]]] = {
        method: [normalize_record(row) for row in rows]
        for method, rows in submissions.items()
    }
    for method, rows in normalized_submissions.items():
        duplicates = duplicate_case_keys(rows)
        if duplicates:
            raise ValueError("%s: duplicate method/sample keys: %s" % (method, duplicates))

    case_sets = {
        method: {str(row["sample_id"]) for row in rows}
        for method, rows in normalized_submissions.items()
    }
    common_cases = set.intersection(*case_sets.values()) if case_sets else set()
    union_cases = set.union(*case_sets.values()) if case_sets else set()
    same_case_contract = all(cases == union_cases for cases in case_sets.values())

    summary_rows: List[Dict[str, Any]] = []
    record_maps = {
        method: {str(row["sample_id"]): row for row in rows}
        for method, rows in normalized_submissions.items()
    }
    contract_fields = ("protocol_id", "benchmark", "group", "subject", "rotation")
    metadata_mismatches: List[Dict[str, Any]] = []
    methods = list(normalized_submissions)
    if methods:
        reference = methods[0]
        for sample_id in sorted(common_cases):
            reference_row = record_maps[reference][sample_id]
            for method in methods[1:]:
                candidate_row = record_maps[method][sample_id]
                differences = {
                    field: {
                        "reference": reference_row[field],
                        "candidate": candidate_row[field],
                    }
                    for field in contract_fields
                    if reference_row[field] != candidate_row[field]
                }
                if differences:
                    metadata_mismatches.append(
                        {
                            "sample_id": sample_id,
                            "reference_method": reference,
                            "method": method,
                            "differences": differences,
                        }
                    )
    matched_contract = same_case_contract and not metadata_mismatches
    # Quality means must use the same successful cases for every method.  If a
    # method fails on one case, keeping that case in another method's mean would
    # reward failures by silently changing the denominator.
    paired_successful_cases = {
        sample_id
        for sample_id in common_cases
        if all(record_maps[method][sample_id]["status"] == "ok" for method in record_maps)
    }
    for method, records in record_maps.items():
        all_rows = list(records.values())
        ok_rows = [row for row in all_rows if row["status"] == "ok"]
        paired_ok = [
            records[sample_id] for sample_id in sorted(paired_successful_cases)
        ]
        for group in ["all"] + sorted({str(row["group"]) for row in paired_ok}):
            selected = paired_ok if group == "all" else [row for row in paired_ok if row["group"] == group]
            for metric in QUALITY_METRICS:
                values = [float(row[metric]) for row in selected if row[metric] is not None]
                summary_rows.append(
                    {
                        "method": method,
                        "group": group,
                        "metric": metric,
                        **describe(values),
                        "submitted_cases": len(all_rows),
                        "successful_cases": len(ok_rows),
                        "success_rate": len(ok_rows) / len(all_rows) if all_rows else 0.0,
                    }
                )

    paired_deltas: List[Dict[str, Any]] = []
    if methods:
        reference = methods[0]
        for method in methods[1:]:
            for metric in QUALITY_METRICS:
                deltas = []
                for sample_id in sorted(common_cases):
                    ref = record_maps[reference][sample_id]
                    candidate = record_maps[method][sample_id]
                    if ref["status"] == candidate["status"] == "ok":
                        deltas.append(float(candidate[metric]) - float(ref[metric]))
                paired_deltas.append(
                    {
                        "reference": reference,
                        "method": method,
                        "metric": metric,
                        "delta_definition": "method_minus_reference; negative_is_better",
                        **describe(deltas),
                    }
                )
    return {
        "method_count": len(submissions),
        "method_case_counts": {method: len(rows) for method, rows in submissions.items()},
        "same_case_contract": same_case_contract,
        "metadata_contract": not metadata_mismatches,
        "matched_contract": matched_contract,
        "metadata_mismatches": metadata_mismatches,
        "union_case_count": len(union_cases),
        "common_case_count": len(common_cases),
        "paired_successful_case_count": len(paired_successful_cases),
        "missing_by_method": {
            method: sorted(union_cases - cases) for method, cases in case_sets.items()
        },
        "summary": summary_rows,
        "paired_deltas": paired_deltas,
        "interpretation_boundary": (
            "No composite score is produced. Quality means use only cases that succeeded for "
            "every submitted method; each method's success rate still uses all submitted cases. "
            "Means are comparable only when the protocol, case set, coordinate contract, and "
            "metric implementation are identical."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    def number(value: Any) -> str:
        return "NA" if value is None else "%.6f" % value

    verdict = "PASS" if report["matched_contract"] else "FAIL"
    lines = [
        "# Cross-method geometry comparison",
        "",
        "**Matched-case contract: %s**" % verdict,
        "",
        "- Methods: %s" % report["method_count"],
        "- Union cases: %s" % report["union_case_count"],
        "- Common cases: %s" % report["common_case_count"],
        "- Metadata mismatches: %s" % len(report["metadata_mismatches"]),
        "- Paired successful cases used for quality means: %s"
        % report["paired_successful_case_count"],
        "- Rule: lower is better for Chamfer, P2S, and normal error.",
        "- No total score is created; each metric answers a different question.",
        "",
        "## Summary on common cases",
        "",
        "| Method | Group | Metric | N | Mean | Median | P90 | Success rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        display = dict(row)
        display.update(
            mean=number(row["mean"]),
            median=number(row["median"]),
            p90=number(row["p90"]),
        )
        lines.append(
            "| {method} | {group} | {metric} | {count} | {mean} | "
            "{median} | {p90} | {success_rate:.2%} |".format(**display)
        )
    if report["paired_deltas"]:
        lines.extend(
            [
                "",
                "## Paired deltas",
                "",
                "Candidate minus reference on the same successful cases; negative is better.",
                "",
                "| Reference | Candidate | Metric | N | Mean delta | Median delta | P90 delta |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["paired_deltas"]:
            display = dict(row)
            display.update(
                mean=number(row["mean"]),
                median=number(row["median"]),
                p90=number(row["p90"]),
            )
            lines.append(
                "| {reference} | {method} | {metric} | {count} | {mean} | "
                "{median} | {p90} |".format(**display)
            )
    if not report["same_case_contract"]:
        lines.extend(["", "## Missing cases", ""])
        for method, values in report["missing_by_method"].items():
            lines.append("- %s: %d missing" % (method, len(values)))
    if report["metadata_mismatches"]:
        lines.extend(["", "## Metadata mismatches", ""])
        for mismatch in report["metadata_mismatches"][:20]:
            fields = ", ".join(sorted(mismatch["differences"]))
            lines.append(
                "- {sample_id}: {reference_method} vs {method} differ in {fields}".format(
                    fields=fields, **mismatch
                )
            )
        if len(report["metadata_mismatches"]) > 20:
            lines.append("- ... %d more" % (len(report["metadata_mismatches"]) - 20))
    lines.extend(["", "## Interpretation boundary", "", report["interpretation_boundary"]])
    return "\n".join(lines) + "\n"


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(markdown_report(report), encoding="utf-8")
    rows = list(report["summary"])
    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare methods under one unified protocol.")
    parser.add_argument(
        "--submission",
        action="append",
        type=parse_submission,
        required=True,
        help="Repeat as METHOD=/path/to/unified.csv. First method is paired reference.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submissions = load_submissions(args.submission)
    report = build_comparison(submissions)
    write_outputs(report, args.output_dir)
    print(
        "COMPARISON_%s methods=%d common_cases=%d report=%s"
        % (
            "PASS" if report["matched_contract"] else "FAIL",
            report["method_count"],
            report["common_case_count"],
            args.output_dir / "comparison.md",
        )
    )
    return 0 if report["matched_contract"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
