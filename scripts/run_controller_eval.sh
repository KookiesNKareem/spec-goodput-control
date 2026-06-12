#!/usr/bin/env bash
# Controller evaluation batch: acceptance-targeting vs goodput controller
# head-to-head, then Poisson arrival QPS sweep. Run after
# run_sharegpt_7b_sweep.sh completes.
set -uo pipefail
cd /workspace
. vllm063/bin/activate
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export TRANSFORMERS_CACHE=/workspace/hf/transformers
export VLLM_LOGGING_LEVEL=ERROR

SPEC="--condition spec_awq --target Qwen/Qwen2.5-7B-Instruct \
  --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ --max-tokens 96 \
  --prompt-source sharegpt --apply-chat-template --no-tqdm"

# --- Head-to-head part 1: goodput controller (probes gamma 1..4 + hard off) ---
echo "=== stage h2h auto ==="
python spec_decode_quant_bench.py $SPEC --spec-tokens 4 \
  --auto-spec-gamma --runtime-hard-off --auto-initial-gamma 1 \
  --num-prompts-list 1,8,16,32 --repeats 7 \
  > spec_clone_7b_sharegpt_auto_hardoff_b1_8_16_32.log 2>&1
echo "stage h2h auto exit=$?"

# --- Head-to-head part 2: acceptance-targeting baseline ---
for t in 0.80 0.90; do
  for b in 1 8 16 32; do
    echo "=== stage h2h acc t=$t b=$b ==="
    python spec_decode_quant_bench.py $SPEC --spec-tokens 4 \
      --acceptance-target "$t" --auto-initial-gamma 1 --auto-window-steps 8 \
      --num-prompts "$b" --repeats 7 \
      > "spec_clone_7b_sharegpt_acctarget${t}_b${b}.log" 2>&1
    echo "stage h2h acc t=$t b=$b exit=$?"
  done
done

# --- Arrival-process QPS sweep ---
for qps in 1 3 6; do
  echo "=== stage arrival target qps=$qps ==="
  python spec_decode_arrival_bench.py --condition target \
    --target Qwen/Qwen2.5-7B-Instruct --qps "$qps" --num-requests 128 \
    --prompt-source sharegpt --apply-chat-template \
    > "arrival_7b_target_qps${qps}.log" 2>&1
  echo "stage arrival target qps=$qps exit=$?"

  echo "=== stage arrival clone g2 qps=$qps ==="
  python spec_decode_arrival_bench.py --condition spec_awq \
    --target Qwen/Qwen2.5-7B-Instruct \
    --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ \
    --qps "$qps" --num-requests 128 --spec-tokens 2 \
    --prompt-source sharegpt --apply-chat-template \
    > "arrival_7b_cloneg2_qps${qps}.log" 2>&1
  echo "stage arrival clone g2 qps=$qps exit=$?"

  echo "=== stage arrival auto qps=$qps ==="
  python spec_decode_arrival_bench.py --condition spec_awq_auto \
    --target Qwen/Qwen2.5-7B-Instruct \
    --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ \
    --qps "$qps" --num-requests 128 --spec-tokens 4 \
    --prompt-source sharegpt --apply-chat-template \
    > "arrival_7b_auto_qps${qps}.log" 2>&1
  echo "stage arrival auto qps=$qps exit=$?"
done

echo "ALL_CONTROLLER_EVAL_DONE"
