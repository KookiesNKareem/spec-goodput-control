#!/usr/bin/env bash
# 3B crossover sweep on ShareGPT prompts: third point for the scaling figure.
set -uo pipefail
cd /workspace
. vllm063/bin/activate
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export TRANSFORMERS_CACHE=/workspace/hf/transformers
export VLLM_LOGGING_LEVEL=ERROR

COMMON="--target Qwen/Qwen2.5-3B-Instruct --num-prompts-list 1,2,4,8,16 \
  --max-tokens 96 --repeats 5 --prompt-source sharegpt --apply-chat-template --no-tqdm"

echo "=== stage 3b target ==="
python spec_decode_quant_bench.py --condition target $COMMON \
  > target_3b_sharegpt_sweep.log 2>&1
echo "stage 3b target exit=$?"

for g in 1 2 3; do
  echo "=== stage 3b clone gamma $g ==="
  python spec_decode_quant_bench.py --condition spec_awq \
    --draft-awq Qwen/Qwen2.5-3B-Instruct-AWQ $COMMON --spec-tokens "$g" \
    > "spec_clone_3b_sharegpt_gamma${g}.log" 2>&1
  echo "stage 3b clone gamma $g exit=$?"
done

echo "ALL_3B_STAGES_DONE"
