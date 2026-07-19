"""Run fixed or adaptive Power Sampling on a small MATH-500 slice."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation import extract_boxed_answer, is_correct
from sampling import (
    AdaptivePowerSampler,
    AdaptivePowerSamplingConfig,
    FixedPowerSampler,
    PowerSamplingConfig,
)
from sampling.hf_backend import HuggingFaceBackend


# Official Qwen2.5-Math CoT system prompt, applied through the chat template.
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--initial-max-new-tokens", type=int, default=2048)
    parser.add_argument("--suffix-max-new-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--gain-threshold", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--rejection-patience", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    from datasets import load_dataset

    backend = HuggingFaceBackend(args.model, args.temperature, args.seed)
    common_config = {
        "steps": args.steps,
        "initial_max_new_tokens": args.initial_max_new_tokens,
        "suffix_max_new_tokens": args.suffix_max_new_tokens,
        "alpha": args.alpha,
        "seed": args.seed,
    }
    if args.mode == "adaptive":
        config = AdaptivePowerSamplingConfig(
            **common_config,
            min_steps=args.min_steps,
            gain_threshold=args.gain_threshold,
            patience=args.patience,
            rejection_patience=args.rejection_patience,
        )
        sampler = AdaptivePowerSampler(backend, config)
    else:
        config = PowerSamplingConfig(**common_config)
        sampler = FixedPowerSampler(backend, config)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").select(
        range(args.limit)
    )
    output_path = args.output or Path(f"results/{args.mode}_power_sampling.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, item in enumerate(dataset):
            started = time.perf_counter()
            problem_seed = args.seed + index
            backend.reseed(problem_seed)
            sampler.reseed(problem_seed)
            prompt = backend.build_chat_prompt(SYSTEM_PROMPT, item["problem"])
            result = sampler.run(prompt)
            initial_answer = extract_boxed_answer(result.initial_text)
            final_answer = extract_boxed_answer(result.final_text)
            record = {
                "index": index,
                "question": item["problem"],
                "ground_truth": item["answer"],
                "initial_answer": initial_answer,
                "initial_is_correct": is_correct(initial_answer, item["answer"]),
                "initial_truncated": result.metrics.initial_tokens
                >= args.initial_max_new_tokens,
                "prompt_style": "qwen-cot-chat",
                "final_answer": final_answer,
                "is_correct": is_correct(final_answer, item["answer"]),
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "model": args.model,
                "mode": args.mode,
                "seed": problem_seed,
                **result.to_dict(),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{index + 1}/{args.limit}] "
                f"initial_correct={record['initial_is_correct']} "
                f"final_correct={record['is_correct']} "
                f"tokens={result.metrics.total_generated_tokens} "
                f"wasted={result.metrics.rejected_tokens} "
                f"saved_steps={result.metrics.saved_attempts}"
            )


if __name__ == "__main__":
    main()
