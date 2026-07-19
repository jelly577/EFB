"""Summarize one or more Power Sampling JSONL result files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, int | float | str]:
    def mean_metric(name: str) -> float:
        return statistics.fmean(record["metrics"][name] for record in records)

    first = records[0]
    return {
        "model": str(first.get("model", "unknown")),
        "mode": str(first.get("mode", "unknown")),
        "problems": len(records),
        "initial_accuracy": statistics.fmean(
            bool(record.get("initial_is_correct", record["is_correct"]))
            for record in records
        ),
        "accuracy": statistics.fmean(bool(record["is_correct"]) for record in records),
        "mean_initial_tokens": mean_metric("initial_tokens"),
        "mean_total_tokens": mean_metric("total_generated_tokens"),
        "mean_rejected_tokens": mean_metric("rejected_tokens"),
        "mean_rejected_token_fraction": mean_metric("rejected_token_fraction"),
        "mean_acceptance_rate": mean_metric("acceptance_rate"),
        "mean_saved_attempts": mean_metric("saved_attempts"),
        "early_stop_rate": statistics.fmean(
            bool(record["metrics"]["stopped_early"]) for record in records
        ),
        "mean_runtime_seconds": statistics.fmean(
            float(record["runtime_seconds"]) for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    summaries = {
        str(path): summarize_records(load_jsonl(path)) for path in args.results
    }
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
