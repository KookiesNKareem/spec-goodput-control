#!/usr/bin/env python3
"""Small vLLM benchmark for draft quantization in speculative decoding."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any


PROMPTS = [
    "Explain why speculative decoding can speed up autoregressive language model inference.",
    "Write a concise Python function that merges two sorted lists without using sorted().",
    "A train travels 90 miles in 1.5 hours. What is its average speed? Show the calculation.",
    "Summarize the difference between post-training quantization and quantization-aware training.",
    "Draft a short email asking a collaborator to review benchmark results by Friday.",
    "What are three likely bottlenecks in serving long-context transformer models?",
    "Given x = 7 and y = 12, compute 3*x + 2*y and explain each step.",
    "List practical reasons why a 4-bit model might not be faster than a 16-bit model.",
    "Translate this sentence to Spanish: The experiment finished earlier than expected.",
    "Give a simple example of a race condition in a multithreaded program.",
    "Why can activation outliers make low-bit quantization difficult?",
    "Write pseudocode for measuring tokens per second for an LLM inference backend.",
    "Compare greedy decoding and temperature sampling in two short paragraphs.",
    "What is the main advantage of using a smaller draft model in speculative decoding?",
    "Create a JSON object with fields name, status, and latency_ms for a completed job.",
    "Explain why measuring wall-clock latency is more useful than only reporting perplexity.",
]


def load_sharegpt_texts(args: argparse.Namespace, tokenizer: Any) -> list[str]:
    import random

    with open(args.prompt_json) as handle:
        records = json.load(handle)
    seen: set[str] = set()
    candidates: list[str] = []
    for record in records:
        conversations = record.get("conversations") or []
        if not conversations:
            continue
        first = conversations[0]
        if first.get("from") != "human":
            continue
        text = (first.get("value") or "").strip()
        if len(text) < 8 or len(text) > args.prompt_max_tokens * 8:
            continue
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    random.Random(args.prompt_seed).shuffle(candidates)
    pool: list[str] = []
    for text in candidates:
        token_count = len(tokenizer.encode(text))
        if args.prompt_min_tokens <= token_count <= args.prompt_max_tokens:
            pool.append(text)
        if len(pool) >= args.prompt_pool_size:
            break
    if len(pool) < args.prompt_pool_size:
        raise ValueError(
            f"only {len(pool)} prompts matched the token bounds; "
            f"need {args.prompt_pool_size}"
        )
    return pool


def build_prompt_pool(args: argparse.Namespace) -> list[str]:
    tokenizer = None
    if args.prompt_source == "sharegpt" or args.apply_chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.target, trust_remote_code=True)
    if args.prompt_source == "builtin":
        texts = list(PROMPTS)
    else:
        texts = load_sharegpt_texts(args, tokenizer)
    if args.apply_chat_template:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in texts
        ]
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["target", "spec_fp", "spec_awq", "spec_ngram"],
        required=True,
    )
    parser.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target-quantization", default=None)
    parser.add_argument("--draft", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--draft-awq", default="Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    parser.add_argument("--draft-awq-quantization", default="awq_marlin")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument(
        "--num-prompts-list",
        default=None,
        help="Comma-separated prompt batch sizes to run after one model load.",
    )
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument(
        "--prompt-source", choices=["builtin", "sharegpt"], default="builtin")
    parser.add_argument("--prompt-json", default="/workspace/sharegpt_v3.json")
    parser.add_argument("--prompt-pool-size", type=int, default=256)
    parser.add_argument("--prompt-min-tokens", type=int, default=16)
    parser.add_argument("--prompt-max-tokens", type=int, default=512)
    parser.add_argument("--prompt-seed", type=int, default=1234)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--spec-tokens", type=int, default=4)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=4)
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=1)
    parser.add_argument(
        "--runtime-spec-tokens",
        type=int,
        default=None,
        help=(
            "Prototype hook: load with --spec-tokens but force each decode "
            "step to use this smaller runtime gamma. Use 0 with "
            "--runtime-hard-off to skip the draft path."
        ),
    )
    parser.add_argument(
        "--batch-gamma-policy",
        default=None,
        help=(
            "Prototype hook: comma-separated '<max_batch>:<gamma>' policy, "
            "for example '24:0,999:2'."
        ),
    )
    parser.add_argument(
        "--runtime-hard-off",
        action="store_true",
        help=(
            "When runtime gamma is 0, bypass the proposer instead of using "
            "vLLM's reversible no-spec path. Safe only when those active "
            "requests will not be re-enabled before finishing."
        ),
    )
    parser.add_argument(
        "--auto-spec-gamma",
        action="store_true",
        help=(
            "Prototype hook: adapt runtime gamma online while loading the "
            "engine with --spec-tokens as the maximum gamma."
        ),
    )
    parser.add_argument("--auto-window-steps", type=int, default=24)
    parser.add_argument("--auto-initial-gamma", type=int, default=1)
    parser.add_argument(
        "--acceptance-target",
        type=float,
        default=None,
        help=(
            "Baseline hook: adapt runtime gamma to hold draft acceptance "
            "near this rate (mimics acceptance-targeting dynamic proposers "
            "such as vLLM PR #26504). Never consults throughput."
        ),
    )
    parser.add_argument("--acceptance-band", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def parse_num_prompts_list(args: argparse.Namespace) -> list[int]:
    if args.num_prompts_list is None:
        prompt_counts = [args.num_prompts]
    else:
        prompt_counts = [
            int(value.strip())
            for value in args.num_prompts_list.split(",")
            if value.strip()
        ]
    if not prompt_counts:
        raise ValueError("at least one prompt count is required")
    if any(prompt_count <= 0 for prompt_count in prompt_counts):
        raise ValueError("prompt counts must be positive")
    return prompt_counts


def main() -> None:
    args = parse_args()
    prompt_counts = parse_num_prompts_list(args)

    os.environ.setdefault("HF_HOME", "/workspace/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/workspace/hf/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf/transformers")

    prompt_pool = build_prompt_pool(args)

    from vllm import LLM, SamplingParams

    def make_prompts(prompt_count: int) -> list[str]:
        return [
            prompt_pool[(args.prompt_offset + index) % len(prompt_pool)]
            for index in range(prompt_count)
        ]

    llm_kwargs = {
        "model": args.target,
        "dtype": args.dtype,
        "seed": args.seed,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": False,
    }
    if args.target_quantization:
        llm_kwargs["quantization"] = args.target_quantization

    if args.condition == "spec_fp":
        llm_kwargs.update(
            speculative_model=args.draft,
            num_speculative_tokens=args.spec_tokens,
            speculative_draft_tensor_parallel_size=1,
        )
    elif args.condition == "spec_awq":
        llm_kwargs.update(
            speculative_model=args.draft_awq,
            speculative_model_quantization=args.draft_awq_quantization,
            num_speculative_tokens=args.spec_tokens,
            speculative_draft_tensor_parallel_size=1,
        )
    elif args.condition == "spec_ngram":
        llm_kwargs.update(
            speculative_model="[ngram]",
            num_speculative_tokens=args.spec_tokens,
            ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=args.ngram_prompt_lookup_min,
        )

    load_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - load_start

    step_adaptive_runtime = bool(
        args.auto_spec_gamma or args.acceptance_target is not None)

    runtime_controller = None
    if args.condition != "target" and (
        args.runtime_spec_tokens is not None
        or args.batch_gamma_policy is not None
        or step_adaptive_runtime
    ):
        from auto_quant_spec_controller import (
            AcceptanceTargetSpecGammaController,
            BatchAdaptiveSpecGammaController,
            BatchPolicySpecGammaController,
            FixedSpecGammaController,
            install_vllm_spec_gamma_controller,
        )

        if args.acceptance_target is not None:
            runtime_controller = AcceptanceTargetSpecGammaController(
                max_gamma=args.spec_tokens,
                initial_gamma=args.auto_initial_gamma,
                target_acceptance=args.acceptance_target,
                band=args.acceptance_band,
                window_steps=args.auto_window_steps,
            )
        elif args.auto_spec_gamma:
            runtime_controller = BatchAdaptiveSpecGammaController(
                max_gamma=args.spec_tokens,
                initial_gamma=args.auto_initial_gamma,
                window_steps=args.auto_window_steps,
                allow_off=args.runtime_hard_off,
                request_level=args.runtime_hard_off,
            )
        elif args.batch_gamma_policy is not None:
            runtime_controller = BatchPolicySpecGammaController.parse(
                args.batch_gamma_policy)
        else:
            runtime_controller = FixedSpecGammaController(args.runtime_spec_tokens)
        install_vllm_spec_gamma_controller(
            llm,
            runtime_controller,
            disable_vllm_spec_metrics=step_adaptive_runtime,
            hard_off=args.runtime_hard_off,
        )

    spec_metrics_collector = None
    if args.condition != "target" and not step_adaptive_runtime:
        worker = getattr(llm.llm_engine.model_executor, "driver_worker", None)
        spec_metrics_collector = getattr(worker, "_metrics", None)
        if spec_metrics_collector is not None:
            spec_metrics_collector._rejsample_metrics_collect_interval_s = 0.0
            spec_metrics_collector._last_metrics_collect_time = 0.0

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    if not args.no_warmup:
        warmup_prompts = make_prompts(1)
        llm.generate(
            warmup_prompts,
            SamplingParams(temperature=0.0, max_tokens=8),
            use_tqdm=not args.no_tqdm,
        )

    capture_logger = None
    if args.condition != "target" and not step_adaptive_runtime:
        from vllm.engine.metrics_types import StatLoggerBase

        class CaptureSpecMetrics(StatLoggerBase):
            def __init__(self) -> None:
                super().__init__(local_interval=0.0)
                self.latest: Any = None

            def log(self, stats: Any) -> None:
                self.maybe_update_spec_decode_metrics(stats)
                if self.spec_decode_metrics is not None:
                    self.latest = self.spec_decode_metrics

            def info(self, type: str, obj: Any) -> None:
                return None

        capture_logger = CaptureSpecMetrics()
        llm.llm_engine.add_logger("capture_spec_metrics", capture_logger)

    last_spec_counts = {
        "accepted_tokens": 0,
        "draft_tokens": 0,
        "emitted_tokens": 0,
    }

    def current_runtime_spec_tokens() -> int | None:
        if args.condition == "target":
            return None
        worker = getattr(llm.llm_engine.model_executor, "driver_worker", None)
        if worker is not None and hasattr(worker, "_auto_spec_last_selected_gamma"):
            selected = getattr(worker, "_auto_spec_last_selected_gamma")
            if selected is not None:
                return int(selected)
        if runtime_controller is not None:
            return int(getattr(runtime_controller, "last_selected_gamma",
                               runtime_controller.current_gamma))
        if args.runtime_spec_tokens is not None:
            return args.runtime_spec_tokens
        return args.spec_tokens

    def count_output_tokens(outputs: Any) -> int:
        return sum(len(request.outputs[0].token_ids) for request in outputs)

    def make_result(
        sweep_index: int,
        repeat: int,
        prompts: list[str],
        elapsed_s: float,
        outputs: Any,
        output_tokens: int,
    ) -> dict[str, Any]:
        runtime_spec_tokens = current_runtime_spec_tokens()
        result: dict[str, Any] = {
            "condition": args.condition,
            "target": args.target,
            "target_quantization": args.target_quantization,
            "draft": None,
            "draft_quantization": None,
            "sweep_index": sweep_index,
            "repeat": repeat,
            "repeats": args.repeats,
            "num_prompts": len(prompts),
            "prompt_offset": args.prompt_offset,
            "prompt_source": args.prompt_source,
            "prompt_seed": args.prompt_seed,
            "apply_chat_template": args.apply_chat_template,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "spec_tokens": args.spec_tokens if args.condition != "target" else None,
            "runtime_spec_tokens": runtime_spec_tokens,
            "runtime_next_spec_tokens": (
                int(getattr(runtime_controller, "current_gamma"))
                if runtime_controller is not None
                else runtime_spec_tokens
            ),
            "auto_spec_gamma": args.auto_spec_gamma,
            "acceptance_target": args.acceptance_target,
            "output_tokens": output_tokens,
            "load_s": load_s,
            "elapsed_s": elapsed_s,
            "output_tok_s": output_tokens / elapsed_s,
        }

        metrics = capture_logger.latest if capture_logger is not None else None
        if metrics is None and spec_metrics_collector is not None:
            event = spec_metrics_collector._copy_rejsample_metrics_async()
            metrics = spec_metrics_collector._collect_rejsample_metrics(
                args.spec_tokens, event
            )

        if metrics is not None:
            current_counts = {
                "accepted_tokens": metrics.accepted_tokens,
                "draft_tokens": metrics.draft_tokens,
                "emitted_tokens": metrics.emitted_tokens,
            }
            if all(
                current_counts[name] >= last_spec_counts[name]
                for name in current_counts
            ):
                delta_counts = {
                    name: current_counts[name] - last_spec_counts[name]
                    for name in current_counts
                }
            else:
                delta_counts = current_counts
            last_spec_counts.update(current_counts)

            draft_tokens = delta_counts["draft_tokens"]
            accepted_tokens = delta_counts["accepted_tokens"]
            emitted_tokens = delta_counts["emitted_tokens"]
            draft_acceptance_rate = (
                accepted_tokens / draft_tokens if draft_tokens else 0.0
            )
            result.update(
                draft_acceptance_rate=draft_acceptance_rate,
                accepted_tokens=accepted_tokens,
                draft_tokens=draft_tokens,
                emitted_tokens=emitted_tokens,
            )
            if not step_adaptive_runtime:
                system_efficiency = (
                    (emitted_tokens / draft_tokens)
                    * (runtime_spec_tokens / (runtime_spec_tokens + 1))
                    if draft_tokens and runtime_spec_tokens
                    else 0.0
                )
                result["system_efficiency"] = system_efficiency

        if args.condition == "spec_fp":
            result["draft"] = args.draft
            result["draft_quantization"] = "fp16"
        elif args.condition == "spec_awq":
            result["draft"] = args.draft_awq
            result["draft_quantization"] = args.draft_awq_quantization
        elif args.condition == "spec_ngram":
            result["draft"] = "[ngram]"
            result["draft_quantization"] = None
        if runtime_controller is not None:
            result["runtime_controller"] = runtime_controller.snapshot()
        return result

    def reset_runtime_request_actions() -> None:
        if not args.runtime_hard_off:
            return
        worker = getattr(llm.llm_engine.model_executor, "driver_worker", None)
        if worker is not None and hasattr(worker, "_auto_spec_request_actions"):
            worker._auto_spec_request_actions = {}

    for sweep_index, prompt_count in enumerate(prompt_counts):
        prompts = make_prompts(prompt_count)
        results = []
        for repeat in range(args.repeats):
            if capture_logger is not None:
                capture_logger.latest = None
            reset_runtime_request_actions()
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=not args.no_tqdm)
            elapsed_s = time.perf_counter() - start
            output_tokens = count_output_tokens(outputs)
            if (
                runtime_controller is not None
                and getattr(runtime_controller, "request_level", False)
            ):
                runtime_controller.record_request(
                    current_runtime_spec_tokens(),
                    len(prompts),
                    output_tokens,
                    elapsed_s * 1000.0,
                )
            result = make_result(
                sweep_index,
                repeat,
                prompts,
                elapsed_s,
                outputs,
                output_tokens,
            )
            results.append(result)
            print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)

        if args.repeats > 1:
            aggregate: dict[str, Any] = {
                "condition": args.condition,
                "target": args.target,
                "target_quantization": args.target_quantization,
                "draft": results[-1]["draft"],
                "draft_quantization": results[-1]["draft_quantization"],
                "sweep_index": sweep_index,
                "num_prompts": len(prompts),
                "prompt_offset": args.prompt_offset,
                "prompt_source": args.prompt_source,
                "prompt_seed": args.prompt_seed,
                "apply_chat_template": args.apply_chat_template,
                "max_tokens": args.max_tokens,
                "spec_tokens": args.spec_tokens if args.condition != "target" else None,
                "runtime_spec_tokens": current_runtime_spec_tokens(),
                "auto_spec_gamma": args.auto_spec_gamma,
                "acceptance_target": args.acceptance_target,
                "repeats": args.repeats,
                "output_tok_s_mean": statistics.mean(
                    result["output_tok_s"] for result in results
                ),
                "output_tok_s_pstdev": statistics.pstdev(
                    result["output_tok_s"] for result in results
                ),
            }
            if all("draft_acceptance_rate" in result for result in results):
                aggregate.update(
                    draft_acceptance_rate_mean=statistics.mean(
                        result["draft_acceptance_rate"] for result in results
                    )
                )
            if all("system_efficiency" in result for result in results):
                aggregate.update(
                    system_efficiency_mean=statistics.mean(
                        result["system_efficiency"] for result in results
                    ),
                )
            if args.repeats > 2:
                steady_results = results[1:]
                aggregate.update(
                    steady_repeats=len(steady_results),
                    output_tok_s_steady_mean=statistics.mean(
                        result["output_tok_s"] for result in steady_results
                    ),
                    output_tok_s_steady_pstdev=statistics.pstdev(
                        result["output_tok_s"] for result in steady_results
                    ),
                )
                if all("draft_acceptance_rate" in result for result in steady_results):
                    aggregate.update(
                        draft_acceptance_rate_steady_mean=statistics.mean(
                            result["draft_acceptance_rate"]
                            for result in steady_results
                        )
                    )
                if all("system_efficiency" in result for result in steady_results):
                    aggregate.update(
                        system_efficiency_steady_mean=statistics.mean(
                            result["system_efficiency"] for result in steady_results
                        ),
                    )
            print("RESULTS_AGG_JSON " + json.dumps(aggregate, sort_keys=True))

    if runtime_controller is not None:
        print(
            "RUNTIME_CONTROLLER_JSON "
            + json.dumps(runtime_controller.snapshot(), sort_keys=True)
        )


if __name__ == "__main__":
    main()
