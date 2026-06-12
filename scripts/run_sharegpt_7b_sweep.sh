#!/usr/bin/env bash
# 7B crossover sweep + baselines on ShareGPT prompts (chat-templated).
# Stages run sequentially after one another; each stage is one engine load.
set -uo pipefail
cd /workspace
. vllm063/bin/activate
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export TRANSFORMERS_CACHE=/workspace/hf/transformers
export VLLM_LOGGING_LEVEL=ERROR

COMMON="--target Qwen/Qwen2.5-7B-Instruct --num-prompts-list 1,4,8,16,24,32 \
  --max-tokens 96 --repeats 5 --prompt-source sharegpt --apply-chat-template --no-tqdm"

echo "=== stage target ==="
python spec_decode_quant_bench.py --condition target $COMMON \
  > target_7b_sharegpt_sweep.log 2>&1
echo "stage target exit=$?"

for g in 1 2 3; do
  echo "=== stage clone gamma $g ==="
  python spec_decode_quant_bench.py --condition spec_awq \
    --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ $COMMON --spec-tokens "$g" \
    > "spec_clone_7b_sharegpt_gamma${g}.log" 2>&1
  echo "stage clone gamma $g exit=$?"
done

echo "=== stage ngram g4 ==="
python spec_decode_quant_bench.py --condition spec_ngram $COMMON --spec-tokens 4 \
  > spec_ngram_7b_sharegpt.log 2>&1
echo "stage ngram exit=$?"

echo "=== stage draft 0.5B g4 ==="
python spec_decode_quant_bench.py --condition spec_awq \
  --draft-awq Qwen/Qwen2.5-0.5B-Instruct-AWQ $COMMON --spec-tokens 4 \
  > spec_draft05b_7b_sharegpt_gamma4.log 2>&1
echo "stage draft 0.5B exit=$?"

echo "=== stage draft 1.5B g4 ==="
python spec_decode_quant_bench.py --condition spec_awq \
  --draft-awq Qwen/Qwen2.5-1.5B-Instruct-AWQ $COMMON --spec-tokens 4 \
  > spec_draft15b_7b_sharegpt_gamma4.log 2>&1
echo "stage draft 1.5B exit=$?"

echo "ALL_STAGES_DONE"
