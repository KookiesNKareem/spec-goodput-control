#!/usr/bin/env bash
# Night 2: V1 off-state tax ablations, then temperature-0.7 and Mistral
# robustness runs on the V0 harness.
set -uo pipefail
cd /workspace
export HF_HOME=/workspace/hf
export HF_HUB_CACHE=/workspace/hf/hub
export TRANSFORMERS_CACHE=/workspace/hf/transformers

# --- Stage A: V1 tax ablations (vllm-v1 venv) ---
. vllm-v1/bin/activate
run_v1() {
  tag="$1"; shift
  echo "=== stage v1 $tag ==="
  python v1_offstate_check.py --tag "$tag" "$@" > "v1_ablate_${tag}.log" 2>&1
  echo "stage v1 $tag exit=$?"
}
run_v1 ngram_k4_idle   --condition ngram --spec-tokens 4 --prompt-lookup-min 15 --prompt-lookup-max 20
run_v1 ngram_k1_active --condition ngram --spec-tokens 1
run_v1 clean_eager     --condition target --enforce-eager
run_v1 ngram_k4_idle_eager --condition ngram --spec-tokens 4 --prompt-lookup-min 15 --prompt-lookup-max 20 --enforce-eager
deactivate

# --- Stage B: temperature 0.7 robustness (vllm063 venv) ---
. vllm063/bin/activate
export VLLM_LOGGING_LEVEL=ERROR
QCOMMON="--target Qwen/Qwen2.5-7B-Instruct --num-prompts-list 1,8,32 \
  --max-tokens 96 --repeats 5 --temperature 0.7 \
  --prompt-source sharegpt --apply-chat-template --no-tqdm"

echo "=== stage t07 target ==="
python spec_decode_quant_bench.py --condition target $QCOMMON \
  > target_7b_sharegpt_t07.log 2>&1
echo "stage t07 target exit=$?"
for g in 1 2 3; do
  echo "=== stage t07 gamma $g ==="
  python spec_decode_quant_bench.py --condition spec_awq \
    --draft-awq Qwen/Qwen2.5-7B-Instruct-AWQ $QCOMMON --spec-tokens "$g" \
    > "spec_clone_7b_sharegpt_t07_gamma${g}.log" 2>&1
  echo "stage t07 gamma $g exit=$?"
done

# --- Stage C: Mistral cross-family check (vllm063 venv) ---
MCOMMON="--target mistralai/Mistral-7B-Instruct-v0.2 --num-prompts-list 1,8,32 \
  --max-tokens 96 --repeats 5 \
  --prompt-source sharegpt --apply-chat-template --no-tqdm"

echo "=== stage mistral target ==="
python spec_decode_quant_bench.py --condition target $MCOMMON \
  > target_mistral7b_sharegpt.log 2>&1
echo "stage mistral target exit=$?"
for g in 1 2; do
  echo "=== stage mistral gamma $g ==="
  python spec_decode_quant_bench.py --condition spec_awq \
    --draft-awq TheBloke/Mistral-7B-Instruct-v0.2-AWQ $MCOMMON --spec-tokens "$g" \
    > "spec_clone_mistral7b_sharegpt_gamma${g}.log" 2>&1
  echo "stage mistral gamma $g exit=$?"
done

echo "ALL_NIGHT2_DONE"
