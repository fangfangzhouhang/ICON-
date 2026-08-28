"""Shared, dependency-free schema for cross-method reconstruction evaluation.

The model-specific runners may use different field names.  They must be
normalised into this schema before validation or comparison so that failures,
missing cases, and metric semantics cannot be hidden by an adapter.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "geometry-reconstruction-v1"
VALID_STATUSES = {"ok", "failed", "skipped"}
QUALITY_METRICS = ("chamfer_cm", "p2s_cm", "normal_error")
SYSTEM_METRICS = (
    "runtime_seconds",
    "peak_memory_allocated_gib",
    "peak_memory_reserved_gib",
)

UNIFIED_FIELDS = (
    "schema_version",
    "protocol_id",
    "benchmark",
    "method",
    "method_variant",
    "sample_id",
    "group",
    "subject",
    "rotation",
    "status",
    "chamfer_cm",
    "p2s_cm",
    "normal_error",
    "runtime_seconds",
    "peak_memory_allocated_gib",
    "peak_memory_reserved_gib",
    "prediction_mesh_path",
    "ground_truth_mesh_path",
    "seed",
    "num_samples",
    "mcube_res",
    "device",
    "git_commit",
    "config_sha256",
    "checkpoint_sha256",
    "normal_checkpoint_sha256",
    "error_type",
    "error_message",
)


def optional_float(value: Any) -> Any:
    """Return a finite float, or ``None`` for an empty value."""
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("must be finite")
    return number


def optional_int(value: Any) -> Any:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def make_sample_id(subject: Any, rotation: Any) -> str:
    subject_text = str(subject).strip()
    rotation_value = optional_int(rotation)
    if not subject_text:
        raise ValueError("subject is required to derive sample_id")
    if rotation_value is None:
        raise ValueError("rotation is required to derive sample_id")
    return "%s@%03d" % (subject_text, rotation_value)


def normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize aliases and value types without inventing metric values."""
    row: Dict[str, Any] = {key: record.get(key, "") for key in UNIFIED_FIELDS}
    row["schema_version"] = str(record.get("schema_version") or SCHEMA_VERSION)
    row["protocol_id"] = str(
        record.get("protocol_id") or "geometry-cape-v1"
    ).strip()
    row["benchmark"] = str(record.get("benchmark") or "cape").strip().lower()
    row["method"] = str(record.get("method") or "").strip()
    row["method_variant"] = str(
        record.get("method_variant") or row["method"]
    ).strip()
    row["group"] = str(record.get("group") or "unspecified").strip()
    row["subject"] = str(record.get("subject") or "").strip()
    row["rotation"] = optional_int(record.get("rotation"))
    row["sample_id"] = str(record.get("sample_id") or "").strip()
    if not row["sample_id"] and row["subject"] and row["rotation"] is not None:
        row["sample_id"] = make_sample_id(row["subject"], row["rotation"])
    row["status"] = str(record.get("status") or "").strip().lower()

    for key in QUALITY_METRICS + SYSTEM_METRICS:
        row[key] = optional_float(record.get(key))
    for key in ("seed", "num_samples", "mcube_res"):
        row[key] = optional_int(record.get(key))

    row["prediction_mesh_path"] = str(
        record.get("prediction_mesh_path")
        or record.get("pred_path")
        or record.get("prediction_mesh")
        or ""
    ).strip()
    row["ground_truth_mesh_path"] = str(
        record.get("ground_truth_mesh_path")
        or record.get("gt_path")
        or record.get("ground_truth_mesh")
        or ""
    ).strip()
    for key in (
        "device",
        "git_commit",
        "config_sha256",
        "checkpoint_sha256",
        "normal_checkpoint_sha256",
        "error_type",
        "error_message",
    ):
        row[key] = str(record.get(key) or "").strip()
    return row


def validate_record(record: Mapping[str, Any], check_paths: bool = False) -> List[str]:
    """Return human-readable validation errors for one canonical record."""
    errors: List[str] = []
    try:
        row = normalize_record(record)
    except (TypeError, ValueError) as exc:
        return ["type conversion failed: %s" % exc]

    for key in ("schema_version", "protocol_id", "benchmark", "method", "sample_id"):
        if not str(row.get(key) or "").strip():
            errors.append("missing required field: %s" % key)
    if row["schema_version"] != SCHEMA_VERSION:
        errors.append(
            "unsupported schema_version: %s (expected %s)"
            % (row["schema_version"], SCHEMA_VERSION)
        )
    if row["status"] not in VALID_STATUSES:
        errors.append("invalid status: %s" % row["status"])

    if row["status"] == "ok":
        for key in QUALITY_METRICS:
            value = row[key]
            if value is None:
                errors.append("successful row missing metric: %s" % key)
            elif value < 0:
                errors.append("metric must be non-negative: %s" % key)
        for key in ("prediction_mesh_path", "ground_truth_mesh_path"):
            if not row[key]:
                errors.append("successful row missing artifact: %s" % key)
    elif row["status"] == "failed" and not (row["error_type"] or row["error_message"]):
        errors.append("failed row must record error_type or error_message")

    if check_paths and row["status"] == "ok":
        for key in ("prediction_mesh_path", "ground_truth_mesh_path"):
            path = Path(row[key])
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append("artifact not found or empty: %s=%s" % (key, path))
    return errors


def case_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    row = normalize_record(record)
    return str(row["method"]), str(row["sample_id"])


def sample_key(record: Mapping[str, Any]) -> str:
    return str(normalize_record(record)["sample_id"])


def duplicate_case_keys(records: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    seen = set()
    duplicates = set()
    for record in records:
        key = case_key(record)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [normalize_record(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    rows = [normalize_record(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(UNIFIED_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in UNIFIED_FIELDS})
