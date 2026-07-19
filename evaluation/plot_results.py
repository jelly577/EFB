"""Create the two plots requested by the project handbook."""

from __future__ import annotations

import argparse
from pathlib import Path

from .summarize_results import load_jsonl, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/power_sampling_report.png"),
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (pareto_axis, position_axis) = plt.subplots(1, 2, figsize=(12, 5))
    ordinary_points: set[tuple[str, float, float]] = set()
    for path in args.results:
        records = load_jsonl(path)
        summary = summarize_records(records)
        label = f"{summary['mode']} ({path.stem})"
        ordinary_key = (
            str(summary["model"]),
            float(summary["mean_initial_tokens"]),
            float(summary["initial_accuracy"]),
        )
        if ordinary_key not in ordinary_points:
            pareto_axis.scatter(
                ordinary_key[1],
                ordinary_key[2],
                s=80,
                marker="x",
                linewidths=2,
                label=f"ordinary ({ordinary_key[0]})",
            )
            ordinary_points.add(ordinary_key)
        pareto_axis.scatter(
            summary["mean_total_tokens"],
            summary["accuracy"],
            s=70,
            label=label,
        )

        normalized_positions = [
            step["split_token_index"] / max(1, step["current_token_count"] - 1)
            for record in records
            for step in record.get("trace", [])
        ]
        if normalized_positions:
            position_axis.hist(
                normalized_positions,
                bins=10,
                range=(0.0, 1.0),
                alpha=0.45,
                label=label,
            )

    pareto_axis.set_xlabel("Mean generated tokens")
    pareto_axis.set_ylabel("Accuracy")
    pareto_axis.set_title("Ordinary vs. Power Sampling")
    pareto_axis.set_ylim(-0.02, 1.02)
    pareto_axis.grid(alpha=0.25)
    pareto_axis.legend()

    position_axis.set_xlabel("Normalized rewrite position")
    position_axis.set_ylabel("Rewrite count")
    position_axis.set_title("Rewrite-position distribution")
    position_axis.grid(alpha=0.25)
    position_axis.legend()

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
