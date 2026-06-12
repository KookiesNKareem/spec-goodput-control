#!/usr/bin/env bash
# 14B crossover sweep on ShareGPT prompts. Weights are ~37.4 GB (FP16 target
# + AWQ clone), so KV is tight: batches capped at 16 and util raised to 0.92.
set -uo pipefail
cd /workspace
. vllm063/bin/activate
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export TRANSFORMERS_CACHE=/workspace/hf/transformers
export VLLM_LOGGING_LEVEL=ERROR

COMMON="--target Qwen/Qwen2.5-14B-Instruct --num-prompts-list 1,4,8,16 \
  --max-tokens 96 --repeats 5 --prompt-source sharegpt --apply-chat-template \
  --no-tqdm --gpu-memory-utilization 0.92"

echo "=== stage 14b target ==="
python spec_decode_quant_bench.py --condition target $COMMON \
  > target_14b_sharegpt_sweep.log 2>&1
echo "stage 14b target exit=$?"

for g in 1 2 3; do
  echo "=== stage 14b clone gamma $g ==="
  python spec_decode_quant_bench.py --condition spec_awq \
    --draft-awq Qwen/Qwen2.5-14B-Instruct-AWQ $COMMON --spec-tokens "$g" \
    > "spec_clone_14b_sharegpt_gamma${g}.log" 2>&1
  echo "stage 14b clone gamma $g exit=$?"
done

echo "ALL_14B_STAGES_DONE"
