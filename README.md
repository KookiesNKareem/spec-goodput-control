# spec-goodput-control

Adaptive speculative decoding for LLM serving, controlled by measured
goodput instead of acceptance rate. Benchmark harness, a runtime controller
prototype for vLLM, and the measurements behind them.

Speculative decoding helps at low load and hurts at high load, but draft
acceptance rate, the signal most controllers watch, barely changes with
load. This repo contains a controller that instead probes speculation
lengths (including *off*) and rewards measured goodput, plus benchmarks
comparing it against acceptance-targeting policies on real workloads.

**Goodput** here means useful work delivered per unit of wall-clock time:
output tokens divided by elapsed seconds, recorded once per completed
request for the speculation setting that served it. It is measured rather
than estimated from a cost model, and it reflects everything that actually
determines serving speed (acceptance, draft cost, verification cost, and
scheduling overhead) instead of any one proxy.

## Key findings

- Draft acceptance is nearly flat across batch sizes (0.86–0.91) while the
  correct action swings from +49% speedup to −22%. No acceptance threshold
  encodes a load-dependent policy.
- The goodput controller matches or beats the fixed-sweep oracle at every
  load and found a better action (k=4 at batch 1, +67%) than the offline
  sweep used to build that oracle.
- Disabling speculation at runtime still costs 15–18% versus an engine
  launched without a draft. This engine-level gap caps every adaptive-k
  controller regardless of policy.
- Under Poisson arrivals, speculation cuts median request latency 19–28% at
  low load; the controller learns per-concurrency policy online and backs
  off at saturation.

Full tables, methodology, and limitations: **[RESULTS.md](RESULTS.md)**.
Raw logs for every number: `results/logs/`.

## Requirements

- One ~46 GB GPU (measurements taken on an NVIDIA A40)
- `vllm==0.6.3.post1`, `torch==2.4.0+cu121`, `transformers==4.46.3`
- Models: Qwen2.5-Instruct 3B/7B/14B and their `-AWQ` variants
- `ShareGPT_V3_unfiltered_cleaned_split.json` for real-workload prompts

## Usage

Fixed-gamma benchmark (target-only, quantized-clone draft, or n-gram):

```bash
python spec_decode_quant_bench.py --condition spec_awq \
  --target Qwen/Qwen2.5-7B-Instruct --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ \
  --spec-tokens 2 --num-prompts-list 1,4,8,16,24,32 --repeats 5 \
  --prompt-source sharegpt --prompt-json ./sharegpt_v3.json --apply-chat-template
```

Adaptive controller (probes k 1..max and hard-off, rewards request goodput):

```bash
python spec_decode_quant_bench.py --condition spec_awq \
  --target Qwen/Qwen2.5-7B-Instruct --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ \
  --spec-tokens 4 --auto-spec-gamma --runtime-hard-off \
  --num-prompts-list 1,8,16,32 --repeats 7 --prompt-source sharegpt --apply-chat-template
```

Acceptance-targeting baseline (the policy shape of vLLM PR #26504):

```bash
python spec_decode_quant_bench.py --condition spec_awq ... \
  --spec-tokens 4 --acceptance-target 0.80 --auto-window-steps 8
```

Open-loop latency under Poisson arrivals:

```bash
python spec_decode_arrival_bench.py --condition spec_awq_auto \
  --target Qwen/Qwen2.5-7B-Instruct --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ \
  --qps 3 --num-requests 128 --spec-tokens 4 --apply-chat-template
```

Summarize logs into per-batch comparison tables:

```bash
python analyze_sweep.py target_7b_sharegpt_sweep.log spec_clone_7b_sharegpt_gamma*.log
```

The exact commands behind every table in RESULTS.md are in `scripts/`.

## Repository structure

| path | contents |
|---|---|
| `spec_decode_quant_bench.py` | batch throughput benchmark; fixed or adaptive runtime speculation |
| `spec_decode_arrival_bench.py` | Poisson arrival benchmark; TTFT / e2e latency percentiles |
| `auto_quant_spec_controller.py` | goodput bandit controller, acceptance-targeting baseline, vLLM 0.6.x runtime hook |
| `analyze_sweep.py` | benchmark logs → comparison tables |
| `scripts/` | sweep runners used for RESULTS.md |
| `patches/` | vLLM 0.6.3 files patched to support mixed FP/AWQ draft checkpoints |
| `results/logs/` | raw benchmark logs |

## Related work

[SmartSpec](https://arxiv.org/abs/2406.14066),
[BanditSpec](https://arxiv.org/abs/2505.15141),
[Cascade](https://arxiv.org/abs/2506.20675),
[QSpec](https://arxiv.org/abs/2410.11305),
[QuantSpec](https://arxiv.org/abs/2502.10424), and vLLM's
[DynamicProposer PR](https://github.com/vllm-project/vllm/pull/26504) /
[adaptive-k issue](https://github.com/vllm-project/vllm/issues/44506).
See [RESULTS.md](RESULTS.md) for how this work relates.

## License

Apache-2.0
