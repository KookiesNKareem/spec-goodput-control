# Runtime Summary

Host: `69.30.85.4:22067`, GPU: NVIDIA A40.

Environment rebuilt at `/workspace/vllm063`.

```text
torch 2.4.0+cu121
transformers 4.46.3
vllm 0.6.3.post1
```

Runtime patches applied in `/workspace/vllm063`:

- AWQ-Marlin reads `modules_to_not_convert` / `ignored_layers`.
- Qwen2 passes linear prefixes and disables quantization at construction time for skipped modules.

These patches are needed for deployable mixed FP/AWQ draft checkpoints.

Models:

- Target: `Qwen/Qwen2.5-3B-Instruct`
- FP draft: `Qwen/Qwen2.5-0.5B-Instruct`
- AWQ draft: `Qwen/Qwen2.5-0.5B-Instruct-AWQ`

## Batch 8, 96 Tokens

| Condition | Gamma | Tok/s | Acceptance | Notes |
|---|---:|---:|---:|---|
| Target only | - | 542.8 | - | baseline |
| FP draft | 4 | 541.5 | 0.692 | tied with target-only |
| AWQ draft | 4 | 580.2 | 0.663 | +6.9% vs target |
| AWQ draft repeat | 4 | 574.0 | 0.663 | confirms positive but some variance |
| VA scale shrink, strength 1.0 | 4 | 458.0 | 0.499 | fails |
| VA scale shrink, strength 0.25 | 4 | 582.7 | 0.663 | within variance of AWQ |
| VA scale expansion, strength -0.25 | 4 | 519.4 | 0.678 | acceptance up, speed down |
| AutoAWQ Calib20 | 4 | 547.0 | 0.668 | same-calibration baseline |
| AutoAWQ Calib20 repeat | 4 | 562.2 | 0.668 | after vLLM mixed-precision patches |
| VA clip override, strength 1.0 | 4 | 545.9 | 0.647 | direct simulated clip transfer fails |
| VA clip bias, strength -0.25 | 4 | 558.5 | 0.667 | throughput variance, no acceptance gain |
| Mixed AWQ protect layers 0,2,17,20 | 4 | 571.0 | 0.680 | deployable mixed FP/AWQ; best same-calibration result |
| Mixed AWQ protect layers 0,2,17,20 repeat | 4 | 548.5 | 0.680 | acceptance stable, tok/s noisy |
| Mixed AWQ protect layers 2,20 | 4 | 498.3 | 0.683 | first run; best acceptance, low tok/s |
| Mixed AWQ protect layers 2,20 repeat | 4 | 550.7 | 0.683 | acceptance stable; throughput above target-only, below best full-AWQ repeat |
| Mixed AWQ protect layers 2,17 | 4 | 521.1 | 0.680 | improves acceptance, worse than 2,20 efficiency |
| Mixed AWQ protect layers 5,14 | 4 | 562.8 | 0.656 | same-cost low-sensitivity control; does not improve acceptance |
| Mixed AWQ protect layer 2 | 4 | 559.0 | 0.676 | single-layer test; layer 2 recovers most of the acceptance gain |
| Mixed AWQ protect layer 20 | 4 | 513.7 | 0.664 | single-layer test; layer 20 alone is not enough |
| FP draft | 6 | 491.6 | 0.724 | slower than target |
| AWQ draft | 6 | 539.5 | 0.696 | roughly target-only |

## Batch 8, 96 Tokens, Repeated Decode After One Load

Same prompts and settings as above, `--repeats 5`. Steady tok/s drops repeat 0.

| Condition | Gamma | Tok/s Mean | Steady Tok/s | Acceptance | System Efficiency | Notes |
|---|---:|---:|---:|---:|---:|---|
| AutoAWQ Calib20 | 4 | 563.1 +/- 2.8 | 564.5 | 0.668 | 0.556 | stable full-AWQ baseline |
| Mixed AWQ protect layer 2 | 4 | 559.2 +/- 8.2 | 563.3 | 0.677 | 0.559 | acceptance up, throughput tied/slightly down |
| Mixed AWQ protect layers 2,20 | 4 | 562.4 +/- 1.2 | 562.8 | 0.684 | 0.563 | best acceptance/efficiency, not faster than full AWQ |
| Mixed AWQ protect layer 2 attention only | 4 | 541.4 +/- 7.3 | 543.6 | 0.669 | 0.554 | deployable but worse; attention is not the layer-2 gain |
| Mixed AWQ protect layer 2 MLP only | 4 | 559.4 +/- 5.6 | 562.2 | 0.674 | 0.560 | recovers most layer-2 gain, still not faster than full AWQ |
| Mixed AWQ protect layer 2 down_proj only | 4 | 559.1 +/- 4.5 | 561.3 | 0.664 | 0.554 | down projection alone does not carry the gain |
| Mixed AWQ protect layer 2 gate/up only | 4 | 559.9 +/- 4.8 | 560.8 | 0.675 | 0.559 | carries MLP acceptance but still slower than full AWQ |
| Mixed AWQ protect layers 2,20 | 5 | 503.2 +/- 0.8 | 503.5 | 0.666 | 0.499 | higher gamma fails; acceptance does not monetize wider speculation |

Attempted a fresh target-only repeat with the same harness, but initialization stalled before GPU load for over six minutes and was stopped. The earlier target-only batch-8 baseline is `542.8 tok/s`.

## Batch 16, 96 Tokens

| Condition | Gamma | Tok/s | Acceptance | Notes |
|---|---:|---:|---:|---|
| Target only | - | 1034.6 | - | target already highly utilized |
| AutoAWQ Calib20 | 4 | 964.0 | 0.653 | slower than target-only |
| Mixed AWQ protect layers 0,2,17,20 | 4 | 944.0 | 0.662 | acceptance up, throughput down |
| Mixed AWQ protect layers 0,2 | 4 | 933.8 | 0.647 | worse than full AutoAWQ |
| Mixed AWQ protect layers 17,20 | 4 | 873.0 | 0.641 | worse than full AutoAWQ |
| Mixed AWQ protect layers 2,20 | 4 | 964.8 | 0.656 | ties full AutoAWQ tok/s with slightly higher acceptance |

## Target Quantization Pivot

Batch 8, 96 tokens, repeated decode after one load:

| Condition | Gamma | Tok/s Mean | Steady Tok/s | Acceptance | System Efficiency | Notes |
|---|---:|---:|---:|---:|---:|---|
| AWQ target only | - | 1151.9 +/- 5.3 | 1153.8 | - | - | `Qwen/Qwen2.5-3B-Instruct-AWQ`, AWQ-Marlin |
| AWQ target + AWQ draft | 4 | 726.1 +/- 9.1 | 730.6 | 0.683 | 0.566 | speculation is much slower once target is quantized |

Target quantization changes the bottleneck: quantizing the 3B verifier gives a large direct speedup, but speculative decoding on top of the quantized target is counterproductive at batch 8 because target scoring is no longer expensive enough to amortize proposal overhead.

Single request, 192 tokens, repeated decode after one load:

| Condition | Gamma | Tok/s Mean | Steady Tok/s | Acceptance | System Efficiency | Notes |
|---|---:|---:|---:|---:|---:|---|
| AWQ target only | - | 156.6 +/- 0.4 | 156.6 | - | - | direct quantized target is >2x the earlier FP target baseline |
| AWQ target + AWQ draft | 4 | 118.8 +/- 0.5 | 119.1 | 0.621 | 0.462 | speculation still slower than quantized target-only |

Target quantization dominates speculative decoding in both tested regimes. This suggests a different quantization-specific speculative question: use a quantized copy of the target as the draft and the FP target as verifier. That may produce much higher acceptance than a smaller draft while still using a faster quantized proposer.

## Quantized Clone As Draft

Setup: FP verifier is `Qwen/Qwen2.5-3B-Instruct`; draft is the same model family quantized as `Qwen/Qwen2.5-3B-Instruct-AWQ` with AWQ-Marlin. This preserves the FP target as the final scorer while using the quantized clone only for proposals.

Batch 8, 96 tokens, repeated decode after one load:

| Condition | Gamma | Tok/s Mean | Steady Tok/s | Acceptance | System Efficiency | Notes |
|---|---:|---:|---:|---:|---:|---|
| FP target only | - | 556.9 +/- 1.7 | 556.6 | - | - | fresh post-rebuild baseline |
| FP target + AWQ 3B clone draft | 1 | 603.8 +/- 6.1 | 606.9 | 0.915 | 0.958 | best so far; +9.0% vs fresh target-only |
| FP target + AWQ 3B clone draft | 2 | 590.4 +/- 1.7 | 590.8 | 0.873 | 0.880 | positive; +6.1% vs fresh target-only |
| FP target + AWQ 3B clone draft | 3 | 545.4 +/- 2.1 | 546.1 | 0.831 | 0.817 | too wide; below target-only |
| FP target + AWQ 3B clone draft | 4 | 489.8 +/- 2.1 | 490.6 | 0.857 | 0.772 | earlier pre-rebuild run; draft/scoring overhead dominates |

Batch 8, 96 tokens, prompt offset 8, repeated decode after one load:

| Condition | Gamma | Tok/s Mean | Steady Tok/s | Acceptance | System Efficiency | Notes |
|---|---:|---:|---:|---:|---:|---|
| FP target only | - | 558.8 +/- 0.3 | 558.9 | - | - | second half of prompt list |
| FP target + AWQ 3B clone draft | 1 | 593.5 +/- 7.0 | 597.0 | 0.882 | 0.941 | +6.8% vs offset-matched target-only |

Batch-size sweep, 96 tokens, prompt offset 0, repeated decode after one load. Raw logs are saved locally as `target_batch_sweep.log` and `spec_clone_gamma1_batch_sweep.log`.

| Batch | Target Steady Tok/s | AWQ Clone Gamma 1 Steady Tok/s | Speedup | Acceptance | System Efficiency | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 74.4 | 81.9 | +10.1% | 0.846 | 0.923 | enable |
| 2 | 144.0 | 161.0 | +11.8% | 0.901 | 0.950 | enable |
| 4 | 284.2 | 317.7 | +11.8% | 0.925 | 0.962 | enable |
| 8 | 559.0 | 602.5 | +7.8% | 0.915 | 0.958 | enable |
| 12 | 810.3 | 807.7 | -0.3% | 0.909 | 0.954 | disable / boundary |
| 16 | 1062.7 | 1031.4 | -2.9% | 0.898 | 0.949 | disable |

7B cross-size validation, 96 tokens, prompt offset 0, repeated decode after one load. Raw logs are saved locally as:

- `target_7b_batch_sweep.log`
- `target_7b_highbatch_sweep.log`
- `target_7b_highbatch_16_24_32.log`
- `spec_clone_7b_gamma1_batch_sweep.log`
- `spec_clone_7b_b12_retry.log`
- `spec_clone_7b_gamma1_highbatch_16_24_32.log`

| Batch | Target Steady Tok/s | AWQ Clone Gamma 1 Steady Tok/s | Speedup | Acceptance | System Efficiency | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 34.5 | 45.7 | +32.5% | 0.959 | 0.980 | enable |
| 2 | 67.0 | 89.6 | +33.7% | 0.910 | 0.955 | enable |
| 4 | 135.1 | 175.9 | +30.3% | 0.924 | 0.962 | enable |
| 8 | 266.6 | 333.7 | +25.2% | 0.939 | 0.970 | enable |
| 12 | 391.4 | 510.6 | +30.4% | 0.932 | 0.966 | enable |
| 16 | 517.9 | 608.7 | +17.5% | 0.912 | 0.956 | enable |
| 24 | 814.8 | 864.9 | +6.1% | 0.920 | 0.960 | enable / boundary |
| 32 | 1026.9 | 1019.4 | -0.7% | 0.910 | 0.955 | disable |

The earlier apparent B12 stall was slow startup, not a failure: the retry completed and gave a strong positive result. The 7B gamma-1 crossover on this A40 is around B24-B32, much later than the 3B crossover around B12-B16. This supports the core scheduler claim that the useful speculation regime expands as verifier cost grows.

7B draft-length sweep, same setup, logs `spec_clone_7b_gamma2_batch_sweep_retry.log`, `spec_clone_7b_gamma3_batch_sweep.log`, and `spec_clone_7b_gamma4_batch_sweep.log`:

| Batch | Target | Gamma 1 | Gamma 2 | Gamma 3 | Gamma 4 | Best Decision | Best Speedup |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 34.5 | 45.7 | 50.0 | 53.6 | 53.5 | gamma 3 | +55.2% |
| 2 | 67.0 | 89.6 | 97.1 | 99.6 | 80.6 | gamma 3 | +48.7% |
| 4 | 135.1 | 175.9 | 185.2 | 189.0 | 158.3 | gamma 3 | +39.9% |
| 8 | 266.6 | 333.7 | 360.9 | 358.9 | 314.7 | gamma 2 | +35.4% |
| 12 | 391.4 | 510.6 | 484.5 | 521.5 | 440.5 | gamma 3 | +33.2% |
| 16 | 517.9 | 608.7 | 595.6 | 569.2 | 543.8 | gamma 1 | +17.5% |
| 24 | 814.8 | 864.9 | - | - | - | gamma 1 | +6.1% |
| 32 | 1026.9 | 1019.4 | - | - | - | off | +0.0% |

This is the first clean method-shaped result: acceptance alone does not choose the right policy. Gamma 1 has the highest acceptance/efficiency, but gamma 2 or gamma 3 produces better throughput at lower and medium batch because extra accepted tokens amortize verifier cost. At high batch, verifier utilization is already high, so the optimal policy shrinks the draft window and then disables speculation.

## Controller Prototype

Implemented a first prototype in `auto_quant_spec_controller.py` and wired it into `spec_decode_quant_bench.py`.

Prototype pieces:

- Offline log parser: reads `RESULTS_AGG_JSON` rows and selects the best action per batch using measured steady tok/s, with a configurable minimum gain threshold.
- Runtime gamma hook: loads vLLM with a maximum `--spec-tokens` value, then monkeypatches the vLLM 0.6.x `SpecDecodeWorker` to choose a smaller per-step `execute_model_req.num_lookahead_slots`.
- Adaptive controller: starts at an initial gamma, probes adjacent gammas over small windows, records per-gamma emitted tokens, stage time, and accepted-position histograms, then chooses the best measured gamma.

Smoke tests:

| Log | Setup | Result |
|---|---|---|
| `runtime_gamma_fixed_smoke.log` | 3B target, AWQ clone draft, engine max gamma 4, forced runtime gamma 1 | Completed; proves max-gamma engine can decode with smaller runtime gamma |
| `runtime_gamma_auto_smoke.log` | 3B target, AWQ clone draft, engine max gamma 3, adaptive gamma, 48 tokens | Completed; controller probed gamma 1/2/3 and ended at gamma 2 on the short run |

Important integration constraint: dynamic gamma `1..max_gamma` works as a prototype because vLLM already passes `num_lookahead_slots` through the speculative worker. Reversible `off` is not clean in vLLM 0.6.x: the no-spec path either keeps the proposer KV cache warm, which still pays draft overhead, or disables a request's speculative state permanently. A production version needs an explicit reversible off mode, likely by tracking draft KV validity and re-prefilling only when speculation is re-enabled.

Single request, 192 tokens, repeated decode after one load:

| Condition | Gamma | Tok/s Mean | Steady Tok/s | Acceptance | System Efficiency | Notes |
|---|---:|---:|---:|---:|---:|---|
| FP target only | - | 75.23 +/- 0.27 | 75.36 | - | - | fresh post-rebuild baseline |
| FP target + AWQ 3B clone draft | 1 | 84.79 +/- 0.10 | 84.81 | 0.856 | 0.928 | +12.5% vs fresh target-only |

Batch 16, 96 tokens, repeated decode after one load:

| Condition | Gamma | Tok/s Mean | Steady Tok/s | Acceptance | System Efficiency | Notes |
|---|---:|---:|---:|---:|---:|---|
| FP target only | - | 1066.5 +/- 0.9 | 1066.8 | - | - | fresh post-rebuild baseline |
| FP target + AWQ 3B clone draft | 1 | 1034.3 +/- 17.8 | 1043.2 | 0.898 | 0.949 | loses at high concurrency; target is already saturated |

Takeaway: quantized-clone speculative decoding has a real speed regime. The object to quantize is the draft, not the verifier: use an AWQ/Marlin copy of the target model as the proposal model and keep the FP target as the verifier. Acceptance is high because the draft is the same model, but the draft is still expensive, so the optimal speculation window is load-dependent. On this A40 with Qwen2.5-3B, gamma 1 is best, gamma 2 still helps, gamma 3/4 fail, and batch 16 loses because the target-only path is already highly utilized. The clean 3B concurrency sweep gives a practical scheduler rule: enable AWQ-clone speculation through batch 8, treat batch 12 as the crossover, and disable by batch 16. The 7B validation is stronger and more method-shaped: the best policy chooses gamma 3 at B1-B4/B12, gamma 2 at B8, gamma 1 at B16-B24, and disables speculation at B32.

## Packed Checkpoint Acceptance Proxy

Benchmark-prompt expected speculative acceptance, 48 target-path contexts, gamma 4:

| Condition | Expected Acceptance | Independent Accept | Top-1 Match | Notes |
|---|---:|---:|---:|---|
| AutoAWQ Calib20 | 0.630 | 0.521 | 0.745 | packed baseline |
| VA clip override, strength 1.0 | 0.624 | 0.486 | 0.740 | agrees with vLLM: worse on benchmark prompts |
| VA clip bias, strength -0.25 | 0.614 | 0.513 | 0.755 | worse |
| AutoAWQ no clip | 0.558 | 0.463 | 0.729 | clipping is not simply the problem |

## Single Request, 192 Tokens

| Condition | Gamma | Tok/s | Acceptance | Notes |
|---|---:|---:|---:|---|
| Target only | - | 75.6 | - | baseline |
| FP draft | 4 | 87.7 | 0.611 | +16.0% vs target |
| AWQ draft | 4 | 85.7 | 0.591 | faster draft, acceptance loss hurts |
| VA scale shrink, strength 0.25 | 4 | 82.3 | 0.563 | fails |
| Mixed AWQ protect layers 2,20 | 4 | 85.7 | 0.591 | no single-request gain over AWQ on this prompt set |

## Takeaway

Standard AWQ draft speculation is a real wall-clock win at moderate batching, and quantized proposal speed matters. But the cheap post-hoc verifier-aware scale perturbation is not robust: strong shrinking fails, weak shrinking is indistinguishable from standard AWQ, and mild expansion improves acceptance while hurting measured throughput.

Direct pre-pack clip transfer is also not robust: applying simulated verifier clip ratios inside AutoAWQ reduces runtime acceptance on benchmark prompts, and disabling clipping is much worse. The first deployed positive result is mixed precision: leave a small verifier-sensitive layer set in FP and pack the rest as AWQ. Protecting layers `0,2,17,20` improves batch-8 acceptance and system efficiency over the same-calibration full-AWQ baseline, but tok/s has run-to-run variance. Protecting `2,20` is a cheaper cross-layer set with the best repeated batch-8 acceptance and the best batch-16 tradeoff among mixed variants tested. Protecting layer `2` alone recovers most of the acceptance gain, protecting `2,17` helps less, the same-cost low-sensitivity control `5,14` hurts acceptance, and protecting layer `20` alone does not recover the gain. The layer choice matters; this is not simply any two FP blocks helping or one late sensitive block dominating.

At batch 16, target-only is already faster than all speculative variants on this A40 setup, so the useful regime is moderate batching or lower concurrency where draft cost matters less than accepted tokens. The method direction that still looks credible is verifier-aware mixed draft quantization with serving-runtime support for skipped modules. The key question is choosing protected layers to maximize throughput, not just acceptance, under the actual target/draft/batch/gamma regime.

Single-request testing weakens the current mixed-protection story: `2,20` does not improve acceptance or throughput versus AWQ on the 1-request prompt. The most credible remaining region is moderate batching, especially batch 8, where layer `2` and `2,20` improve acceptance/system efficiency but tok/s needs cleaner repeated measurement.

Repeated batch-8 measurement confirms the current mixed-protection method is not yet a throughput win over full AWQ. It does improve verifier acceptance and system efficiency in a stable way (`2,20`: acceptance `0.684` vs `0.668`; efficiency `0.563` vs `0.556`), but the added FP draft cost cancels the gain in output tok/s. This is useful as a negative/steering result: the selection objective must be cost-aware, and acceptance-only layer protection is too weak as a speed method.

Submodule protection narrows the phenomenon: layer-2 attention-only protection is deployable but hurts throughput and does not improve acceptance, while layer-2 MLP-only protection recovers most of the full layer-2 acceptance/efficiency gain with less code-surface area. However, even MLP-only does not beat full AWQ tok/s at gamma 4. Increasing the speculation window to gamma 5 for `2,20` fails badly, so the added acceptance does not monetize wider speculation. Down-proj-only does not carry the MLP gain; gate/up-only carries the MLP acceptance gain but is still slower than full AWQ. This mixed-protection path is likely not strong enough as a speed contribution.

The stronger method direction is now quantized-clone speculation: same-size AWQ draft, FP verifier, and a cost-aware scheduler that chooses whether to activate speculation and how many draft tokens to propose. The immediate rule from current evidence is model/load dependent: for Qwen2.5-3B on A40, use gamma 1 at low/moderate batch and disable by B16; for Qwen2.5-7B on A40, use gamma 3 at low/mid batch, gamma 1 near high-batch boundary, and disable around B32.

## Adaptive Runtime Controller

Implemented `auto_quant_spec_controller.py` and wired it into `spec_decode_quant_bench.py`.

The prototype loads vLLM with a maximum speculation window and chooses a smaller runtime gamma per decode step by monkeypatching vLLM 0.6.x `SpecDecodeWorker`. The controller keeps separate bandit state by active-batch bucket, probes gamma values, and now uses wall-clock step goodput as the primary reward. The earlier stage-time-only reward was useful for debugging but mispredicted high-batch throughput.

7B AWQ-clone controller results, 96 generated tokens, 4 repeats, steady mean drops repeat 0:

| Batch | Target tok/s | Fixed-gamma oracle | Stage-reward adaptive | Wall-reward adaptive | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 34.5 | gamma 3, 53.6 | 53.2 | - | Recovers ~99% of oracle |
| 8 | 266.6 | gamma 2, 360.9 | 369.5 | - | Recovers oracle within run variance |
| 16 | 517.9 | gamma 1, 608.7 | 571.6 | 615.7 | Wall reward fixes high-batch choice |
| 24 | 814.8 | gamma 1, 864.9 | 747.5 | 867.0 | Wall reward fixes failure and matches oracle |

This is the cleanest proof so far. Dynamic quantized-clone speculation can recover the fixed-sweep optimum without a user-provided offline table, but the controller must optimize observed serving goodput rather than acceptance or internal speculative-stage timing alone. The user-facing method should be a low-overhead online controller integrated into the serving runtime: keep the verifier exact, quantize the clone draft, probe a small gamma set, and choose by measured tokens per wall-clock decode step for the current active-batch/load state.

Remaining production gap: vLLM 0.6.x does not have a clean reversible `off` state. Gamma `1..max_gamma` can be controlled dynamically today, but true online enable/disable needs explicit draft-KV validity handling so the controller can turn speculation off at saturated load and re-enable it later without permanently disabling requests or paying proposer work while "off".

## Hard-Off Prototype

Added an explicit hard-off path to `auto_quant_spec_controller.py` and a benchmark flag in `spec_decode_quant_bench.py`.

The new path distinguishes:

- Soft off: vLLM's default no-spec path. It emits target tokens but still runs the proposer to keep draft KV warm.
- Hard off: when runtime gamma is `0`, monkeypatch `_run_no_spec` to skip the proposer. This gives the speed benefit of off, but it is safe only if those active requests are not re-enabled before finishing.
- Request-level reversible off: hard-off a high-batch request, then re-enable speculation for a later fresh request in the same loaded engine.

3B smoke test, same A40, 64 generated tokens, 3 repeats, steady mean drops repeat 0:

| Condition | Batch | Steady tok/s | Runtime gamma | Notes |
|---|---:|---:|---:|---|
| Target only | 16 | 1049.2 | - | exact target baseline |
| Spec engine, soft off | 16 | 714.8 | 0 | proposer still runs to keep draft KV warm |
| Spec engine, hard off | 16 | 926.5 | 0 | skips draft compute; ~30% faster than soft off |
| Same spec engine after hard-off batch | 4 | 270.9 | 1 | speculation re-enabled for fresh requests; acceptance 0.910 |
| Target only | 4 | 282.1 | - | same baseline run |

This proves the practical request-level mechanism: a loaded speculative engine can skip draft work at high batch and later use the draft again for new requests. The prototype includes a conservative stale-request guard: once an active request has skipped proposer execution, that request is forced to remain hard-off until it finishes. It does not yet solve active-sequence re-enable after skipped proposer steps; that requires a draft-KV catch-up/replay mechanism.

Request-level hard-off controller result, 7B AWQ clone, 96 generated tokens, 7 repeats. Raw log: `spec_clone_7b_auto_hardoff_requestlevel_reqreward_b8_16_24_32.log`.

The important correction was reward level. A step-level wall-clock EWMA incorrectly overvalued hard-off because it compared individual decode steps and tail effects, not completed request goodput. The fixed controller records one completed-request reward after each `generate()` call: `output_tokens / elapsed_ms` for the action that actually ran. The forced probe order is gamma 1, gamma 2, gamma 3, off; repeats 4-6 show the settled choice.

| Batch | Probe gamma 1 | Probe gamma 2 | Probe gamma 3 | Probe off | Settled Action | Settled Tok/s | Target/Oracle Context |
|---:|---:|---:|---:|---:|---|---:|---|
| 8 | 333.8 | 373.8 | 373.6 | 264.5 | gamma 2 | 374.3 | target 266.6; fixed oracle gamma 2/3 around 360 |
| 16 | 623.5 | 598.5 | 569.6 | 506.8 | gamma 1 | 622.0 | target 517.9; fixed oracle gamma 1 around 608.7 |
| 24 | 873.9 | 858.8 | 790.8 | 789.9 | gamma 1 | 871.5 | target 814.8; fixed oracle gamma 1 around 864.9 |
| 32 | 1028.4 | 971.8 | 957.8 | 988.1 | gamma 1 | 1008.6 | target 1026.9; gamma 1/off are near boundary |

This is the current strongest controller proof. It uses no offline user table, tries off as a real action, rejects off when it is bad, and recovers the right gamma in the positive regimes. At B32 it still chooses gamma 1 because the measured request-level reward was slightly above hard-off in this run, but the target-only baseline remains marginally better; this reinforces the production controller design: include a true non-speculative target-only arm or a stronger high-load margin so saturated regimes do not pay speculative-engine overhead for negligible gain.

## Dynamic Load-Shift Demo

Added a cost-aware tie-break to the controller: after probing, choose the cheapest action whose measured request-level goodput is within the improvement margin of the fastest measured action. The cheapness order is `off`, then narrower gamma values. This keeps probe overhead low while avoiding wider speculation for tiny/noisy wins.

Load-shift run: one loaded 7B speculative engine, phases `B8 -> B24 -> B32 -> B16 -> B8`, 96 generated tokens, 7 repeats per phase. Raw log: `spec_clone_7b_loadshift_costaware_b8_24_32_16_8.log`.

| Phase | Batch | Action Sequence | Settled Action | Settled Tok/s | Notes |
|---:|---:|---|---:|---:|---|
| 0 | 8 | 1,2,3,0,3,3,3 | 3 | 355.2 | learns low-batch speculation; off is rejected |
| 1 | 24 | 1,2,3,0,2,2,2 | 2 | 860.5 | adapts to measured phase reward; gamma 2 was clearly ahead in this run |
| 2 | 32 | 1,2,3,0,1,1,1 | 1 | 993.0 | high-load boundary; gamma 1 beats hard-off inside spec engine, but target-only baseline is still competitive |
| 3 | 16 | 1,2,3,0,1,1,1 | 1 | 617.3 | cost-aware tie-break avoids picking gamma 2 for a tiny probe lead |
| 4 | 8 | 3,3,3,3,3,3,3 | 3 | 371.8 | reuses learned B8 policy immediately; no re-probe needed |

The demo supports the integration claim: the runtime can self-tune as load changes, with no model-specific offline lookup table supplied by the user. It also exposes the right production refinement: one-sample probes are noisy, so action selection should be conservative and cost-aware, with periodic re-probes or confidence windows for long-running servers.

## ShareGPT Real-Workload Sweep (7B)

Setup: ShareGPT V3 first human turns, deduplicated, 16-512 prompt tokens, Qwen
chat template applied, pool 256, seed 1234 (`--prompt-source sharegpt
--apply-chat-template`). 96 generated tokens, 5 repeats per batch after one
load, steady mean drops repeat 0. Logs: `target_7b_sharegpt_sweep.log`,
`spec_clone_7b_sharegpt_gamma{1,2,3}.log`, `spec_ngram_7b_sharegpt.log`.

| Batch | Target | Clone g1 | Clone g2 | Clone g3 | Ngram g4 | Oracle |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 33.8 | 41.7 (+23.4%) | 45.4 (+34.2%) | 50.6 (+49.6%) | 32.1 (-4.9%) | gamma 3 |
| 4 | 131.9 | 149.3 (+13.2%) | 154.4 (+17.0%) | 125.2 (-5.1%) | 116.6 (-11.6%) | gamma 2 |
| 8 | 252.3 | 260.3 (+3.2%) | 277.3 (+9.9%) | 213.5 (-15.4%) | 226.7 (-10.2%) | gamma 2 |
| 16 | 461.5 | 419.5 (-9.1%) | 470.5 (+1.9%) | 384.9 (-16.6%) | 390.9 (-15.3%) | off/boundary |
| 24 | 703.0 | 589.0 (-16.2%) | 599.9 (-14.7%) | 491.9 (-30.0%) | 507.4 (-27.8%) | off |
| 32 | 867.8 | 733.5 (-15.5%) | 677.9 (-21.9%) | 589.8 (-32.0%) | 654.6 (-24.6%) | off |

Findings:

1. The real-workload crossover is much earlier than on the handwritten prompt
   set: gamma-1 crossover moves from B24-32 down to B8-16. Offline tables built
   on synthetic prompts mispredict real serving on the same hardware and model.
2. Clone acceptance is nearly flat in batch (0.85-0.91 for gamma 1/2 at every
   batch), while the correct action swings from +49.6% (enable gamma 3 at B1)
   to -22% (disable at B32). Acceptance-rate signals cannot drive this policy.
   Across gammas, acceptance also fails as a gradient: gamma 3 has acceptance
   0.885 at B1 (+49.6%) and 0.780 at B8 (-15.4%).
3. Ngram/prompt-lookup speculation loses at every batch on chat workloads
   (acceptance 0.22-0.35).
4. Small-draft baselines (0.5B/1.5B AWQ) cannot load against the 7B target:
   vLLM asserts on vocab mismatch (152064 vs 151936). Within Qwen2.5, no
   smaller same-vocab draft exists for 7B+ targets; the quantized clone is the
   only in-family, training-free draft. Logs:
   `spec_draft05b_7b_sharegpt_gamma4.log` (assertion in
   `spec_decode_worker.py::_vocab_size`).

## 14B ShareGPT Crossover (Scaling Point)

Same protocol as the 7B ShareGPT sweep; `--gpu-memory-utilization 0.92`,
batches capped at 16 (37.4 GB of weights). Logs:
`target_14b_sharegpt_sweep.log`, `spec_clone_14b_sharegpt_gamma{1,2,3}.log`.

| Batch | Target | Clone g1 | Clone g2 | Clone g3 | Oracle |
|---:|---:|---:|---:|---:|---|
| 1 | 18.9 | 24.0 (+27.0%) | 23.4 (+23.9%) | 28.6 (+51.6%) | gamma 3 |
| 4 | 71.3 | 89.8 (+25.9%) | 93.2 (+30.7%) | 87.1 (+22.2%) | gamma 2 |
| 8 | 138.7 | 166.8 (+20.2%) | 171.9 (+23.9%) | 167.6 (+20.8%) | gamma 2 |
| 16 | 251.4 | 251.5 (+0.1%) | 233.9 (-7.0%) | 231.7 (-7.8%) | off |

Every positive-batch win is larger at 14B than 7B with the same oracle shape
(g3 at B1, g2 mid-range, off at the top); the g1 crossover moves from B8-16
out to ~B16. Supports the scaling claim: speculation regime widens with
verifier cost on real workloads.

## 3B ShareGPT Crossover (Scaling Point)

Same protocol, batches 1-16. Logs: `target_3b_sharegpt_sweep.log`,
`spec_clone_3b_sharegpt_gamma{1,2,3}.log` (gamma 3 rerun after a disk-quota
failure killed the first attempt).

| Batch | Target | Clone g1 | Clone g2 | Clone g3 | Oracle |
|---:|---:|---:|---:|---:|---|
| 1 | 73.8 | 78.2 (+6.0%) | 84.7 (+14.8%) | 83.1 (+12.6%) | gamma 2 |
| 2 | 140.5 | 154.5 (+10.0%) | 160.0 (+13.9%) | 159.7 (+13.7%) | gamma 2 |
| 4 | 271.8 | 286.3 (+5.3%) | 295.5 (+8.7%) | 275.4 (+1.3%) | gamma 2 |
| 8 | 533.8 | 519.2 (-2.7%) | 520.5 (-2.5%) | 512.0 (-4.1%) | off |
| 16 | 952.6 | 751.0 (-21.2%) | 723.3 (-24.1%) | 663.8 (-30.3%) | off |

Three-point scaling on one workload: crossover ~B4-8 at 3B, ~B8-16 at 7B,
~B16 at 14B; peak win grows +14.8% -> +49.6% -> +51.6%. The optimal action
differs at every (model, load) point.

## Controller Head-to-Head on ShareGPT (7B)

Goodput controller: `--auto-spec-gamma --runtime-hard-off`, max gamma 4, probe
order gamma 1,2,3,4,off, 7 repeats, settled = mean of last 3. Acceptance
baseline: `--acceptance-target {0.80,0.90} --auto-window-steps 8`, the policy
shape of vLLM PR #26504 (widen when acceptance above band, narrow when below;
no off action; never consults throughput). Logs:
`spec_clone_7b_sharegpt_auto_hardoff_b1_8_16_32.log`,
`spec_clone_7b_sharegpt_acctarget{0.80,0.90}_b{1,8,16,32}.log`.

| Batch | Acc-target 0.80 | Acc-target 0.90 | Goodput controller | Reference |
|---:|---:|---:|---:|---|
| 1 | 49.2 (dithers g1-g3) | 46.4 (pinned g1/g2) | 56.4 (g4) | best fixed g3: 50.6 |
| 8 | 270.2 (dithers) | 273.4 (g1-heavy) | 272.4 (g1) | best fixed g2: 277.3 |
| 16 | 472.3 (mostly g2) | 461.3 (g1/g2) | 445.0 (g1) | best fixed g2: 470.5 |
| 32 | 682.7 (g1) | 735.0 (g1) | 715.1 (off) | true target-only: 867.8 |

Findings:

1. The goodput controller discovered gamma 4 at B1 (56.4 tok/s, +67% vs
   target), beating the fixed-gamma oracle table itself, which only swept
   gamma 1-3. Measured exploration found an action the offline table missed.
2. No acceptance setpoint works at both ends of the load curve: 0.80 dithers
   mid-gammas, 0.90 pins narrow. Both leave 13-18% at B1 and neither can
   disable at B32 (acceptance EWMA 0.83 looks healthy while speculation
   burns throughput). Acceptance oscillates around any threshold rather than
   settling; per-repeat actions show persistent dithering.
3. Goodput controller loss at B16 (445.0 vs 472.3): one-sample probe noise at
   a boundary load (g2 probed 451 vs g1 443; steady g2 is ~470). Production
   fix: confidence-windowed or repeated probes.
4. Spec-engine tax, quantified: the best in-engine action at B32 (g1 or
   hard-off, 715-735 tok/s) is 15-18% below a clean target-only engine
   (867.8). vLLM 0.6.x pessimizes the whole engine when a draft is configured
   (async output processing disabled, spec scheduling path). Worker-level
   hard-off cannot recover this; dynamic control needs engine-level
   reversible off. This applies equally to acceptance-targeting proposers.

## Arrival-Process Latency (7B, Poisson, ShareGPT)

`spec_decode_arrival_bench.py`: open-loop Poisson arrivals, 128 requests, 96
generated tokens, in-process vLLM 0.6.x engine. Conditions: target-only engine,
fixed clone gamma 2, adaptive goodput controller (max gamma 4, hard-off).
Logs: `arrival_7b_{target,cloneg2,auto}_qps{1,3,6}.log`.

| QPS | Condition | e2e p50 | e2e p90 | e2e p99 | TTFT p50 | tok/s | max running |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | target | 3.06 | 3.21 | 3.31 | 0.066 | 107.4 | 10 |
| 1 | clone g2 | 2.21 | 2.48 | 2.91 | 0.066 | 108.2 | 8 |
| 1 | adaptive | 2.15 | 2.93 | 3.40 | 0.062 | 108.2 | 8 |
| 3 | target | 3.36 | 3.48 | 3.60 | 0.065 | 306.2 | 20 |
| 3 | clone g2 | 2.71 | 3.11 | 3.41 | 0.061 | 310.8 | 17 |
| 3 | adaptive | 2.72 | 3.38 | 4.33 | 0.073 | 307.8 | 17 |
| 6 | target | 3.86 | 4.12 | 4.35 | 0.070 | 571.4 | 34 |
| 6 | clone g2 | 5.11 | 6.13 | 7.15 | 0.078 | 574.7 | 42 |
| 6 | adaptive | 4.52 | 5.30 | 6.61 | 0.076 | 560.6 | 36 |

Findings:

1. Low/mid load: speculation converts to real request latency, -28%/-19% p50
   at QPS 1/3 with identical offered throughput. The adaptive controller
   matches fixed-best p50 without prior knowledge of the workload, and at
   QPS 1 it learned the per-concurrency policy online (gamma 4 at
   concurrency <=1, gamma 2 at 2-8) matching the fixed-batch oracle.
2. High load (QPS 6): target-only wins (3.86 p50). The controller correctly
   backs off and beats fixed speculation (4.52 vs 5.11 p50; 6.61 vs 7.15
   p99) but cannot reach target parity due to the spec-engine tax.
3. Controller probe overhead lands in the tail (p99 3.40 vs 2.91 at QPS 1);
   same root cause as the B16 head-to-head miss, same fix
   (confidence-windowed probes).

## V1 Spot-Check (vLLM 0.22.1)

Same A40 and ShareGPT workload, batch 32, `v1_offstate_check.py`: clean
engine 1081.3 tok/s steady; ngram speculative config attached 919.7 tok/s
(-14.9%). V1 exposes no runtime off action, so the tax is unavoidable for
any adaptive controller. Confirms the V0 off-state finding on current main.
