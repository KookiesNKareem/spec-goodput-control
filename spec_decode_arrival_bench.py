#!/usr/bin/env python3
"""Open-loop arrival benchmark for quantized-clone speculative decoding.

Submits requests to an in-process vLLM 0.6.x engine with Poisson arrivals at a
configured QPS and measures per-request TTFT and end-to-end latency, plus
aggregate goodput. Conditions mirror spec_decode_quant_bench.py: target-only,
fixed-gamma AWQ clone speculation, or the adaptive goodput controller with
request-level hard-off.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["target", "spec_awq", "spec_awq_auto"],
        required=True,
    )
    parser.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-awq", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--draft-awq-quantization", default="awq_marlin")
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--arrival-seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--spec-tokens", type=int, default=3)
    parser.add_argument("--runtime-spec-tokens", type=int, default=None)
    parser.add_argument("--auto-window-steps", type=int, default=24)
    parser.add_argument("--auto-initial-gamma", type=int, default=1)
    parser.add_argument(
        "--prompt-source", choices=["builtin", "sharegpt"], default="sharegpt")
    parser.add_argument("--prompt-json", default="/workspace/sharegpt_v3.json")
    parser.add_argument("--prompt-pool-size", type=int, default=256)
    parser.add_argument("--prompt-min-tokens", type=int, default=16)
    parser.add_argument("--prompt-max-tokens", type=int, default=512)
    parser.add_argument("--prompt-seed", type=int, default=1234)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1,
                max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def main() -> None:
    args = parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/workspace/hf/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf/transformers")

    from spec_decode_quant_bench import build_prompt_pool

    prompt_pool = build_prompt_pool(args)

    from vllm import EngineArgs, LLMEngine, SamplingParams

    engine_kwargs: dict[str, Any] = {
        "model": args.target,
        "dtype": args.dtype,
        "seed": args.seed,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
    }
    if args.condition in ("spec_awq", "spec_awq_auto"):
        engine_kwargs.update(
            speculative_model=args.draft_awq,
            speculative_model_quantization=args.draft_awq_quantization,
            num_speculative_tokens=args.spec_tokens,
            speculative_draft_tensor_parallel_size=1,
        )

    load_start = time.perf_counter()
    engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
    load_s = time.perf_counter() - load_start

    runtime_controller = None
    worker = None
    if args.condition == "spec_awq_auto" or (
        args.condition == "spec_awq" and args.runtime_spec_tokens is not None
    ):
        from auto_quant_spec_controller import (
            BatchAdaptiveSpecGammaController,
            FixedSpecGammaController,
            install_vllm_spec_gamma_controller,
        )

        if args.condition == "spec_awq_auto":
            runtime_controller = BatchAdaptiveSpecGammaController(
                max_gamma=args.spec_tokens,
                initial_gamma=args.auto_initial_gamma,
                window_steps=args.auto_window_steps,
                allow_off=True,
                request_level=True,
            )
        else:
            runtime_controller = FixedSpecGammaController(
                args.runtime_spec_tokens)
        install_vllm_spec_gamma_controller(
            SimpleNamespace(llm_engine=engine),
            runtime_controller,
            disable_vllm_spec_metrics=args.condition == "spec_awq_auto",
            hard_off=args.condition == "spec_awq_auto",
        )
        worker = getattr(engine.model_executor, "driver_worker", None)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Warm up so the first arrival does not pay one-time CUDA graph costs.
    engine.add_request("warmup", prompt_pool[0],
                       SamplingParams(temperature=0.0, max_tokens=8))
    while engine.has_unfinished_requests():
        engine.step()

    rng = random.Random(args.arrival_seed)
    arrival_offsets: list[float] = []
    elapsed = 0.0
    for _ in range(args.num_requests):
        elapsed += rng.expovariate(args.qps)
        arrival_offsets.append(elapsed)

    arrivals: dict[str, float] = {}
    first_token: dict[str, float] = {}
    finishes: dict[str, float] = {}
    output_tokens: dict[str, int] = {}
    max_running = 0

    start = time.perf_counter()
    next_index = 0
    while len(finishes) < args.num_requests:
        now = time.perf_counter() - start
        while (
            next_index < args.num_requests
            and arrival_offsets[next_index] <= now
        ):
            request_id = str(next_index)
            prompt = prompt_pool[next_index % len(prompt_pool)]
            engine.add_request(request_id, prompt, params)
            arrivals[request_id] = now
            next_index += 1

        if not engine.has_unfinished_requests():
            if next_index < args.num_requests:
                wait_s = arrival_offsets[next_index] - (
                    time.perf_counter() - start)
                if wait_s > 0:
                    time.sleep(min(wait_s, 0.005))
            continue

        step_outputs = engine.step()
        step_now = time.perf_counter() - start
        running = sum(1 for output in step_outputs if not output.finished)
        max_running = max(max_running, running)
        for output in step_outputs:
            request_id = output.request_id
            if request_id == "warmup" or request_id not in arrivals:
                continue
            token_count = len(output.outputs[0].token_ids)
            if token_count > 0 and request_id not in first_token:
                first_token[request_id] = step_now
            if output.finished:
                finishes[request_id] = step_now
                output_tokens[request_id] = token_count
                if runtime_controller is not None and worker is not None:
                    actions = getattr(worker, "_auto_spec_request_actions", {})
                    gamma = actions.get(request_id)
                    if gamma is not None and hasattr(runtime_controller,
                                                     "record_request"):
                        elapsed_ms = (step_now - arrivals[request_id]) * 1000.0
                        runtime_controller.record_request(
                            int(gamma), running + 1, token_count, elapsed_ms)

    makespan_s = max(finishes.values()) - min(arrivals.values())
    e2e = sorted(finishes[rid] - arrivals[rid] for rid in finishes)
    ttft = sorted(
        first_token[rid] - arrivals[rid] for rid in first_token
        if rid in finishes
    )
    total_tokens = sum(output_tokens.values())

    result = {
        "condition": args.condition,
        "target": args.target,
        "draft": args.draft_awq if args.condition != "target" else None,
        "qps": args.qps,
        "num_requests": args.num_requests,
        "arrival_seed": args.arrival_seed,
        "max_tokens": args.max_tokens,
        "spec_tokens": (
            args.spec_tokens if args.condition != "target" else None),
        "runtime_spec_tokens": args.runtime_spec_tokens,
        "prompt_source": args.prompt_source,
        "prompt_seed": args.prompt_seed,
        "apply_chat_template": args.apply_chat_template,
        "load_s": load_s,
        "makespan_s": makespan_s,
        "output_tok_s": total_tokens / makespan_s,
        "max_observed_running": max_running,
        "e2e_mean_s": statistics.mean(e2e),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p90_s": percentile(e2e, 0.90),
        "e2e_p99_s": percentile(e2e, 0.99),
        "ttft_mean_s": statistics.mean(ttft) if ttft else 0.0,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p90_s": percentile(ttft, 0.90),
        "ttft_p99_s": percentile(ttft, 0.99),
    }
    if runtime_controller is not None:
        result["runtime_controller"] = runtime_controller.snapshot()
    print("ARRIVAL_RESULT_JSON " + json.dumps(result, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
