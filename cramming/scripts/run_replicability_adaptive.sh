#!/usr/bin/env bash
set -euo pipefail

# Adaptive runner for replicability experiments.
# Simplified policy: big steps while >0.9, dense 10-token steps
# between 0.9 and 0.1, then one extra big step after <0.1 and stop.
#
# Usage:
#   bash cramming/scripts/run_replicability_adaptive.sh
# Adjust variables below as needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRAMMING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS=(
  "EleutherAI/pythia-160m"
  "EleutherAI/pythia-410m"
  "EleutherAI/pythia-1b"
  "Qwen/Qwen2.5-0.5B"
  "Qwen/Qwen2.5-1.5B"
  "meta-llama/Llama-3.2-1B"
  "google/gemma-3-270m"
  "google/gemma-3-1b-pt"
)

MEMS=(1 2 3 4 5)
SAMPLES=20 # number of drawn strings per $n$ to estimate average accessibility
REPEATS=1 # number of cramming attempts before considering it non accessible
ITERS=3000 # steps of cramming 
DEVICE="cuda" 
SHUFFLED="true" #  "true" = random, false = pg19 
BASE_SEED=1337

# Adaptive sweep parameters
MIN_LENGTH=10
MAX_LENGTH=1000
BIG_STEP=100
SMALL_STEP=10
TEXTS_PATH="${CRAMMING_DIR}/data/pg19_valid_1k_chunks.csv"
VOCAB_PATH="${CRAMMING_DIR}/data/vocab_100k.txt"
SAVE_DIR="${CRAMMING_DIR}/runs"

# Single run over all MEMS; results/logs go to runs/<model_name>
SEED=$((BASE_SEED))
for MODEL in "${MODELS[@]}"; do
  ARGS=(
    --model_name "${MODEL}"
    --N_mem_tokens "${MEMS[@]}"
    --num_samples "${SAMPLES}"
    --num_repeats "${REPEATS}"
    --num_iterations "${ITERS}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --early_stopping_patience 500
    --adaptive
    --lr 1e-2
    --min_length "${MIN_LENGTH}"
    --max_length "${MAX_LENGTH}"
    --big_step "${BIG_STEP}"
    --small_step "${SMALL_STEP}"
    --dtype "float16"
    --texts_path "${TEXTS_PATH}"
    --vocab_path "${VOCAB_PATH}"
    --save_dir "${SAVE_DIR}"
  )
  if [[ "${SHUFFLED}" == "true" ]]; then
    ARGS+=(--shuffled)
  fi
  if [[ "${DEVICE}" == cuda* ]]; then
    ARGS+=(--use_flash_attention_2)
  fi

  echo "[adaptive job] model=${MODEL} mems=[${MEMS[*]}] shuffled=${SHUFFLED} min=${MIN_LENGTH} max=${MAX_LENGTH} big=${BIG_STEP} small=${SMALL_STEP}"
  python "${CRAMMING_DIR}/replicability.py" "${ARGS[@]}"
done

echo "Adaptive job completed."
