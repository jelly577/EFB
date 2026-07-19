"""Run the fixed Power Sampling baseline on a small MATH-500 slice."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation import extract_boxed_answer, is_correct
from sampling import FixedPowerSampler, PowerSamplingConfig
from sampling.hf_backend import HuggingFaceBackend


PROMPT_TEMPLATE = (
    "Solve the following math problem. Think step by step and put the final answer "
    "in \\boxed{{}}.\n\nProblem: {problem}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--initial-max-new-tokens", type=int, default=1024)
    parser.add_argument("--suffix-max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fixed_power_sampling.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    from datasets import load_dataset

    backend = HuggingFaceBackend(args.model, args.temperature, args.seed)
    config = PowerSamplingConfig(
        steps=args.steps,
        initial_max_new_tokens=args.initial_max_new_tokens,
        suffix_max_new_tokens=args.suffix_max_new_tokens,
        seed=args.seed,
    )
    sampler = FixedPowerSampler(backend, config)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").select(
        range(args.limit)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for index, item in enumerate(dataset):
            started = time.perf_counter()
            prompt = PROMPT_TEMPLATE.format(problem=item["problem"])
            result = sampler.run(prompt)
            final_answer = extract_boxed_answer(result.final_text)
            record = {
                "index": index,
                "question": item["problem"],
                "ground_truth": item["answer"],
                "final_answer": final_answer,
                "is_correct": is_correct(final_answer, item["answer"]),
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "model": args.model,
                **result.to_dict(),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{index + 1}/{args.limit}] "
                f"correct={record['is_correct']} "
                f"tokens={result.metrics.total_generated_tokens} "
                f"wasted={result.metrics.rejected_tokens}"
            )


if __name__ == "__main__":
    main()
