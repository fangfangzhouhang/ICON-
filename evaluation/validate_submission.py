"""Validate one model submission before any cross-paper comparison."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from evaluation.schema import (
    duplicate_case_keys,
    load_csv,
    normalize_record,
    sample_key,
    validate_record,
)


def expected_sample_ids(path: Path) -> Set[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected: Set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id and row.get("subject") and row.get("rotation") not in (None, ""):
            sample_id = "%s@%03d" % (row["subject"].strip(), int(float(row["rotation"])))
        if not sample_id:
            raise ValueError("manifest row is missing sample_id or subject+rotation")
        if sample_id in expected:
            raise ValueError("manifest contains duplicate sample_id: %s" % sample_id)
        expected.add(sample_id)
    return expected


def build_report(
    records: List[Mapping[str, Any]],
    expected: Optional[Set[str]] = None,
    expected_count: Optional[int] = None,
    check_paths: bool = False,
) -> Dict[str, Any]:
    normalized = [normalize_record(record) for record in records]
    row_errors = []
    for index, record in enumerate(normalized, start=2):
        errors = validate_record(record, check_paths=check_paths)
        if errors:
            row_errors.append(
                {"csv_line": index, "sample_id": record.get("sample_id"), "errors": errors}
            )

    duplicates = ["%s:%s" % key for key in duplicate_case_keys(normalized)]
    observed = {sample_key(record) for record in normalized if record.get("sample_id")}
    missing = sorted((expected or set()) - observed)
    unexpected = sorted(observed - (expected or observed))
    statuses: Dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0, "invalid": 0}
    for record in normalized:
        status = str(record.get("status") or "invalid")
        statuses[status] = statuses.get(status, 0) + 1

    count_matches = expected_count is None or len(normalized) == expected_count
    contract_valid = (
        not row_errors
        and not duplicates
        and not missing
        and not unexpected
        and count_matches
    )
    success_rate = statuses.get("ok", 0) / len(normalized) if normalized else 0.0
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "contract_valid": contract_valid,
        "record_count": len(normalized),
        "unique_sample_count": len(observed),
        "expected_sample_count": None if expected is None else len(expected),
        "declared_expected_count": expected_count,
        "count_matches": count_matches,
        "status_counts": statuses,
        "success_rate": success_rate,
        "duplicate_case_keys": duplicates,
        "missing_sample_ids": missing,
        "unexpected_sample_ids": unexpected,
        "row_errors": row_errors,
        "path_checks_enabled": check_paths,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    verdict = "PASS" if report["contract_valid"] else "FAIL"
    lines = [
        "# Unified evaluation submission validation",
        "",
        "**Verdict: %s**" % verdict,
        "",
        "- Records: %s" % report["record_count"],
        "- Unique samples: %s" % report["unique_sample_count"],
        "- Expected samples: %s" % report["expected_sample_count"],
        "- Declared expected count: %s" % report["declared_expected_count"],
        "- Count matches declaration: %s" % report["count_matches"],
        "- Success rate: %.2f%%" % (100.0 * report["success_rate"]),
        "- Status counts: `%s`" % report["status_counts"],
        "- Path checks enabled: %s" % report["path_checks_enabled"],
        "",
        "Failures are kept in the denominator. A failed model case is valid evidence; "
        "a malformed or silently missing case is not.",
    ]
    for title, key in (
        ("Duplicate method/sample keys", "duplicate_case_keys"),
        ("Missing expected samples", "missing_sample_ids"),
        ("Unexpected samples", "unexpected_sample_ids"),
    ):
        values = report[key]
        if values:
            lines.extend(["", "## %s" % title, ""] + ["- `%s`" % value for value in values])
    if report["row_errors"]:
        lines.extend(["", "## Row errors", ""])
        for item in report["row_errors"]:
            lines.append(
                "- line %s, `%s`: %s"
                % (item["csv_line"], item["sample_id"], "; ".join(item["errors"]))
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a unified evaluation CSV.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Fail when the CSV does not contain exactly this many rows.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Also require local prediction and GT mesh files to exist and be non-empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_csv(args.input)
    expected = expected_sample_ids(args.expected_manifest) if args.expected_manifest else None
    report = build_report(
        records,
        expected=expected,
        expected_count=args.expected_count,
        check_paths=args.check_paths,
    )
    output_dir = args.output_dir or args.input.parent / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        "VALIDATION_%s records=%d success_rate=%.2f%% report=%s"
        % (
            "PASS" if report["contract_valid"] else "FAIL",
            report["record_count"],
            report["success_rate"] * 100.0,
            output_dir / "validation.md",
        )
    )
    return 0 if report["contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
