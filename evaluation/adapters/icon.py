"""Convert ``run_cape_pilot`` CSV output into the shared result contract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from evaluation.adapters.base import ResultAdapter
from evaluation.schema import SCHEMA_VERSION, normalize_record, write_csv


class IconCapeAdapter(ResultAdapter):
    def convert(self, source: Path) -> List[Dict[str, Any]]:
        with source.open("r", newline="", encoding="utf-8-sig") as stream:
            raw_rows = list(csv.DictReader(stream))

        converted: List[Dict[str, Any]] = []
        for raw in raw_rows:
            row: Dict[str, Any] = dict(raw)
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": "geometry-cape-v1",
                    "benchmark": "cape",
                    "method_variant": raw.get("method") or "icon-filter",
                    "seed": raw.get("seed"),
                    "num_samples": raw.get("num_samples") or 1000,
                    "git_commit": raw.get("git_commit") or "",
                }
            )
            converted.append(normalize_record(row))
        return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ICON CAPE case CSV to the shared evaluation schema."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = IconCapeAdapter().convert(args.input)
    write_csv(args.output, records)
    success = sum(row["status"] == "ok" for row in records)
    print(
        "ICON_ADAPTER_OK rows=%d success=%d failed_or_skipped=%d output=%s"
        % (len(records), success, len(records) - success, args.output)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

