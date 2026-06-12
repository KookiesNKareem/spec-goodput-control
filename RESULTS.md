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
disabling speculation matters most. Engine-level reversible off, with
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
and its speedup vs target-only, same workload and hardware:

| batch | 7B best | 7B speedup | 14B best | 14B speedup |
|---|---|---|---|---|
| 1 | k3 | +49.6% | k3 | +51.6% |
| 4 | k2 | +17.0% | k2 | +30.7% |
| 8 | k2 | +9.9% | k2 | +23.9% |
| 16 | off | (k2 +1.9%) | off | (k1 +0.1%) |
| 24 | off | — | | |
| 32 | off | — | | |

At every positive batch size the 14B win is larger than the 7B win, with the
same oracle shape (wide k at batch 1, k=2 mid-range, off at the top). The
k=1 crossover moves from batch 8–16 (7B) to ~16 (14B). A 3B column is being
added (logs will appear in `results/logs/`); on a synthetic prompt set the
3B crossover sat at batch 12–16 vs 7B's 24–32, the same ordering.

Side finding: for Qwen2.5 ≥7B there is no smaller same-vocab family member,
so classic small-draft speculation cannot even load (vLLM asserts in
`spec_decode_worker.py::_vocab_size`, 152064 vs 151936). The quantized clone
is the only in-family training-free draft. N-gram/prompt-lookup speculation
loses at every batch size on this workload (acceptance 0.22–0.35, −5% to
−28% throughput).

## Limitations

vLLM 0.6.x (V0 engine) prototype; single GPU type; greedy decoding; fixed
96-token outputs; the off-state tax number is engine-specific even though
the gap itself is general; one-sample probes are noisy at boundary loads.
