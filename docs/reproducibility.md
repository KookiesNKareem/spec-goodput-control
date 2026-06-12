# Reproducibility Notes

This repo records a research workflow around quantized-clone speculative
decoding. The scripts are meant to run inside a GPU workspace, not on a laptop.

## Tested Environment

The main results in `runtime_summary.md` used:

```text
Host: 69.30.85.4:22067
GPU: NVIDIA A40, 46 GB
Workspace: /workspace
Python environment: /workspace/vllm063
torch: 2.4.0+cu121
transformers: 4.46.3
vllm: 0.6.3.post1
```

The V1 off-state spot-check used a separate `vllm-v1` environment with vLLM
`0.22.1`.

## Data

ShareGPT runs expect a JSON file at:

```text
/workspace/sharegpt_v3.json
```

The benchmark loader keeps first human turns, deduplicates text, filters prompts
to 16-512 tokenizer tokens by default, shuffles with seed `1234`, and applies
the target model chat template when `--apply-chat-template` is set.

## vLLM Patch Files

Two local patch copies are kept under `patches/`:

```text
patches/awq_marlin.py
patches/qwen2.py
```

They were applied to vLLM `0.6.3.post1` so AWQ-Marlin could honor
`modules_to_not_convert` / `ignored_layers` and Qwen2 construction could skip
quantization for selected module prefixes. This matters for the mixed FP/AWQ
draft experiments. The quantized-clone ShareGPT sweeps do not depend on the
mixed-layer checkpoint generation path, but they were run in the same patched
environment.

Find the installed package paths with:

```bash
python - <<'PY'
import inspect
import pathlib
import vllm

root = pathlib.Path(inspect.getfile(vllm)).parent
print(root / "model_executor/layers/quantization/awq_marlin.py")
print(root / "model_executor/models/qwen2.py")
PY
```

Then replace those installed package files with the repo copies inside the
benchmark virtualenv.

## Environment Setup

For the vLLM 0.6.3 serving benchmarks:

```bash
cd /workspace
python -m venv vllm063
. vllm063/bin/activate
pip install -r /path/to/spec-goodput-control/requirements-serving-vllm063.txt

export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export TRANSFORMERS_CACHE=/workspace/hf/transformers
export VLLM_LOGGING_LEVEL=ERROR
```

Optional older quantization scripts require additional packages such as
AutoAWQ, safetensors, tqdm, and huggingface_hub. Those scripts are retained for
provenance of the negative verifier-aware quantization experiments, not for the
main serving-control claim.

## Main Runs

The runners assume they are launched from this repo but immediately `cd` into
`/workspace`. Copy or clone the repo contents into `/workspace`, or adjust the
runner paths before launching.

```bash
bash scripts/run_sharegpt_3b_sweep.sh
bash scripts/run_sharegpt_7b_sweep.sh
bash scripts/run_sharegpt_14b_sweep.sh
bash scripts/run_controller_eval.sh
```

Additional robustness checks:

```bash
bash scripts/run_robustness_checks.sh
```

That batch runs:

- V1 off-state ablations.
- Qwen2.5-7B ShareGPT at temperature 0.7.
- Mistral-7B cross-family sanity checks.

## Result Extraction

New raw logs are ignored by default because they are generated artifacts; the
published logs under `results/logs/` are intentionally tracked. When logs are
present locally, collect them with:

```bash
python3 scripts/collect_results.py results/logs/*.log \
  --json-out results/current-results.json \
  --markdown-out results/current-results.md
```

For fixed-batch sweeps, `analyze_sweep.py` prints a compact Markdown table from
`RESULTS_AGG_JSON` lines:

```bash
python3 analyze_sweep.py \
  results/logs/target_7b_sharegpt_sweep.log \
  results/logs/spec_clone_7b_sharegpt_gamma1.log \
  results/logs/spec_clone_7b_sharegpt_gamma2.log \
  results/logs/spec_clone_7b_sharegpt_gamma3.log \
  results/logs/spec_ngram_7b_sharegpt.log
```

## Interpreting Results

Use steady means when a benchmark reports both full-run and steady throughput.
The convention in this repo is to drop repeat 0 for fixed-batch sweeps because
the first repeat often includes one-time runtime effects after model load.

The controller head-to-head uses completed-request reward:

```text
output tokens / request wall-clock time
```

Step-level timing was tested and rejected because it overvalued hard-off through
tail effects and did not match completed-request goodput.
