# POSTED 2026-06-12 to PR #26504 (venue changed from issue #44506 per Kareem)
# https://github.com/vllm-project/vllm/pull/26504#issuecomment-4686567840

Glad to see per-request dynamic k getting built, the variable-length proposal plumbing here is exactly what any adaptive policy needs. I recently benchmarked this PR's policy shape (EWMA acceptance with a target band) and want to share data that might be useful for the signal design, plus a related gap.

Setup: Qwen2.5-7B FP16 target with an AWQ copy of itself as the draft, A40, ShareGPT prompts, greedy, prototyped on vLLM 0.6.3 (harness, controller, and raw logs: https://github.com/KookiesNKareem/spec-goodput-control).

The acceptance-threshold signal struggled on this workload: acceptance is nearly flat across load (0.86-0.91 from batch 1 to 32) while the right action flips from +23% (speculate at b=1) to -16% (disable at b=32). I ran acceptance-targeting at two setpoints against picking k by measured goodput (output tokens / wall-time per finished request, probing k in {1..4} plus off):

| batch | acc-target 0.80 | acc-target 0.90 | goodput | target-only |
|---|---|---|---|---|
| 1 | 49.2 tok/s | 46.4 | **56.4** (k4) | 33.8 |
| 32 | 682.7 (k1) | 735.0 (k1) | **715.1** (off) | **867.8** |

Neither setpoint works at both ends: acceptance dips at wide k, so they never hold k=4 at b=1, and they can't disable at b=32 where acceptance still reads a healthy 0.83. The goodput probe found k=4 at b=1 (+67%), which my own offline sweep had missed. So it might be worth considering measured goodput (or utility, cf. #44506) as the adaptation signal on top of this PR's plumbing.

The related gap: "off" isn't free. Even with the proposer fully skipped, a spec-configured engine loses 15-18% vs a clean engine at b=32. Same check on main (0.22.1): 1081 tok/s clean vs 920 with an ngram spec config attached (-15%), and V1 currently has no runtime off at all. Until off is a real action with "off = target-only" as the bar, that caps every adaptive-k policy, including this one.

Happy to help test this PR, or to prototype a goodput-based signal on top of it.
