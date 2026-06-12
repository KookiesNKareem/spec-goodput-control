# Results

All numbers: Qwen2.5 targets (FP16), AWQ-Marlin quantized copy of the target
as draft, vLLM 0.6.3.post1, single NVIDIA A40, ShareGPT first-turn prompts
with chat template (pool 256, seed 1234), greedy decoding, 96 output tokens.
Batch tables use 5–7 repeats after one engine load; "settled" = mean of the
last 3 repeats. Raw logs for every table are in `results/logs/`.

## 1. Acceptance is flat in load; the right action is not

Draft acceptance vs batch size, 7B target, speculation window k=1:

| batch | 1 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| acceptance | 0.900 | 0.859 | 0.900 | 0.909 | 0.911 | 0.913 |
| speedup vs target-only | +23% | +13% | +3% | −9% | −16% | −16% |

A controller watching acceptance sees the same signal at batch 1 (+23%
available) and batch 32 (−16% if it speculates). On this workload the signal
does not vary with the outcome of the decision it is supposed to drive.

## 2. Head-to-head: goodput reward vs acceptance-targeting

The acceptance-targeting baseline implements the policy shape of
[vLLM PR #26504](https://github.com/vllm-project/vllm/pull/26504): EWMA of
observed acceptance; widen k above a target band, narrow below; no off
action. The goodput controller probes k∈{1..4}∪{off} and rewards output
tokens per wall-clock second per completed request. 7B target:

| batch | acc-target 0.80 | acc-target 0.90 | goodput controller | best fixed | target-only |
|---|---|---|---|---|---|
| 1 | 49.2 (dithers k1↔k3) | 46.4 (pins k1/k2) | **56.4 (k4)** | k3: 50.6 | 33.8 |
| 8 | 270.2 | 273.4 | 272.4 (k1) | k2: 277.3 | 252.3 |
| 16 | 472.3 | 461.3 | 445.0 (k1) | k2: 470.5 | 461.5 |
| 32 | 682.7 (k1) | 735.0 (k1) | **715.1 (off)** | — | **867.8** |

- No acceptance setpoint works at both ends of the load curve. Both leave
  13–18% on the table at batch 1 (per-position acceptance dips below any
  reasonable threshold at wide k, so they never hold k=4) and neither
  disables at batch 32, where acceptance still reads 0.83.
- The goodput controller discovered k=4 at batch 1 (+67% vs target-only);
  the offline sweep used to build the "best fixed" column only tested k≤3.
- Honest miss: at batch 16, one-sample probes picked k1 over k2 (−5%).
  Probe windows need confidence accumulation.

## 3. The off-state tax: 15–18%

The batch-32 row above is the headline infrastructure finding. Even with the
proposer fully skipped (hard-off: `_run_no_spec` patched to bypass draft
execution), the spec-configured engine tops out at 715–735 tok/s where a
clean engine without `speculative_model` reaches 867.8. In vLLM 0.6.x the
gap comes from engine-level pessimizations applied whenever a draft is
configured (async output processing disabled, speculative scheduling path).

Consequence: every adaptive-k controller (acceptance-based, utility-based
like [Cascade](https://arxiv.org/abs/2506.20675), or goodput-based like
this repo) is capped well below target-only exactly in the regime where
disabling speculation matters most.

Spot-checked on current vLLM (0.22.1, V1 engine), same hardware and
workload at batch 32: a clean engine reaches 1081 tok/s steady; the same
engine with an ngram speculative config attached reaches 920 tok/s, a
14.9% loss (`v1_offstate_check.py`, logs in `results/logs/`). The V1
number bundles ngram proposal cost with engine overhead since V1 exposes
no runtime off action at all, which is precisely the gap: a controller
that wants to stop speculating under load currently has no mechanism that
recovers target-only performance. Engine-level reversible off, with
"off ≈ target-only" as the bar, is a prerequisite for this feature class.

## 4. Latency under Poisson arrivals

Open-loop arrivals, 128 requests (`spec_decode_arrival_bench.py`), 7B:

| QPS | condition | e2e p50 | e2e p99 | tok/s |
|---|---|---|---|---|
| 1 | target-only | 3.06 s | 3.31 s | 107 |
| 1 | clone k=2 | 2.21 s | 2.91 s | 108 |
| 1 | adaptive | **2.15 s** | 3.40 s | 108 |
| 3 | target-only | 3.36 s | 3.60 s | 306 |
| 3 | adaptive | 2.72 s | 4.33 s | 308 |
| 6 | target-only | **3.86 s** | 4.35 s | 571 |
| 6 | clone k=2 | 5.11 s | 7.15 s | 575 |
| 6 | adaptive | 4.52 s | 6.61 s | 561 |

Speculation converts to real request latency where it should (−28% / −19%
median at QPS 1/3). The adaptive controller matches fixed-best median while
learning the per-concurrency policy online (k=4 at concurrency ≤1, k=2 at
2–8), and at QPS 6 it correctly backs off, but cannot reach target parity
because of the off-state tax (§3). Controller probe overhead lands in the
p99 tail; same root cause and fix as the §2 batch-16 miss.

## 5. Model-size scaling

The useful speculation regime widens with verifier cost. Best clone action
and its speedup vs target-only, same workload and hardware, three model
sizes:

| batch | 3B | 7B | 14B |
|---|---|---|---|
| 1 | k2 +14.8% | k3 +49.6% | k3 +51.6% |
| 4 | k2 +8.7% | k2 +17.0% | k2 +30.7% |
| 8 | off (best spec −2.5%) | k2 +9.9% | k2 +23.9% |
| 16 | off (−21%) | off (k2 +1.9%) | off (k1 +0.1%) |
| 24 | | off (−15%) | |
| 32 | | off (−16%) | |

The crossover moves out monotonically with verifier size: roughly batch 4–8
at 3B, 8–16 at 7B, and 16 at 14B, and the peak win grows from +15% to +52%.
The optimal policy differs at every (model, load) point, which is the case
for learning it online rather than shipping a table.

Side finding: for Qwen2.5 ≥7B there is no smaller same-vocab family member,
so classic small-draft speculation cannot even load (vLLM asserts in
`spec_decode_worker.py::_vocab_size`, 152064 vs 151936). The quantized clone
is the only in-family training-free draft. N-gram/prompt-lookup speculation
loses at every batch size on this workload (acceptance 0.22–0.35, −5% to
−28% throughput).

## 6. Robustness checks: sampling and another model family

Additional A40 runs on 2026-06-12 keep the same policy shape under
temperature sampling and with Mistral-7B-Instruct-v0.2. Full run notes are in
`docs/runs/2026-06-12-a40.md`.

Qwen2.5-7B, ShareGPT, temperature=0.7:

| batch | target | clone k1 | clone k2 | clone k3 | best action |
|---|---:|---:|---:|---:|---|
| 1 | 34.0 | 41.4 (+21.7%) | 45.0 (+32.4%) | 43.3 (+27.3%) | k2 |
| 8 | 252.3 | 291.0 (+15.4%) | 305.1 (+20.9%) | 275.0 (+9.0%) | k2 |
| 32 | 870.3 | 723.1 (−16.9%) | 686.5 (−21.1%) | 652.6 (−25.0%) | off |

Mistral-7B-Instruct-v0.2, ShareGPT, greedy:

| batch | target | clone k1 | clone k2 | best action |
|---|---:|---:|---:|---|
| 1 | 35.6 | 46.9 (+31.8%) | 55.4 (+55.5%) | k2 |
| 8 | 257.9 | 300.2 (+16.4%) | 315.6 (+22.3%) | k2 |
| 32 | 835.3 | 692.9 (−17.1%) | 763.4 (−8.6%) | off |

In both checks the acceptance signal remains high where the goodput decision
should turn speculation off: Qwen temperature sampling has k1 acceptance
0.884 at batch 32 while losing 16.9%, and Mistral k2 acceptance is 0.889
while losing 8.6%.

## Limitations

vLLM 0.6.x (V0 engine) prototype; single GPU type; greedy decoding; fixed
96-token outputs; the off-state tax number is engine-specific even though
the gap itself is general; one-sample probes are noisy at boundary loads.
