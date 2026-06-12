#!/usr/bin/env python3
"""Spot-check the speculative off-state cost on current vLLM (V1 engine).

Measures batch-32 decode throughput for a clean engine vs the same engine
with a speculative config attached, to test whether the V0 finding (a
spec-configured engine pays a large tax even when speculation contributes
nothing) still holds on the V1 engine.

Run separately per condition so each engine gets a clean process:
  python v1_offstate_check.py --condition target
  python v1_offstate_check.py --condition ngram --spec-tokens 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["target", "ngram"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--spec-tokens", type=int, default=4)
    parser.add_argument("--prompt-lookup-max", type=int, default=4)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt-json", default="/workspace/sharegpt_v3.json")
    parser.add_argument("--prompt-pool-size", type=int, default=256)
    parser.add_argument("--prompt-min-tokens", type=int, default=16)
    parser.add_argument("--prompt-max-tokens", type=int, default=512)
    parser.add_argument("--prompt-seed", type=int, default=1234)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/workspace/hf/hub")

    args.prompt_source = "sharegpt"
    args.apply_chat_template = True
    args.target = args.model
    from spec_decode_quant_bench import build_prompt_pool

    prompts = build_prompt_pool(args)[: args.num_prompts]

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.condition == "ngram":
        llm_kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }

    llm = LLM(**llm_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            ignore_eos=True)
    llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=8),
                 use_tqdm=False)

    tok_s = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tok_s.append(tokens / elapsed)

    import vllm

    result = {
        "vllm_version": vllm.__version__,
        "condition": args.condition,
        "tag": args.tag,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "enforce_eager": args.enforce_eager,
        "prompt_lookup": (
            [args.prompt_lookup_min, args.prompt_lookup_max]
            if args.condition == "ngram" else None),
        "spec_tokens": args.spec_tokens if args.condition == "ngram" else None,
        "tok_s_all": [round(v, 1) for v in tok_s],
        "tok_s_steady_mean": round(statistics.mean(tok_s[1:]), 1),
    }
    print("V1_CHECK_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
