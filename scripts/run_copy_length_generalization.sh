#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-${LLM_VIS_MODEL_ROOT:-${REPO_ROOT}/models/llms-theory}}"
MODEL_LIST="${MODEL_LIST:-${REPO_ROOT}/models_list.csv}"
MODELS="${MODELS:-EleutherAI/pythia-160m,EleutherAI/pythia-410m}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/copying/synthetic_parent_results}"

TRAIN_MIN_LEN="${TRAIN_MIN_LEN:-5}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-50}"
EVAL_MIN_LEN="${EVAL_MIN_LEN:-10}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-1000}"
EVAL_STEP="${EVAL_STEP:-5}"
CONTEXT_LEN="${CONTEXT_LEN:-2048}"

LR="${LR:-1e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-6}"
IN_DIST_NUM_BATCHES="${IN_DIST_NUM_BATCHES:-6}"
TRAIN_STEPS_PER_CHECK="${TRAIN_STEPS_PER_CHECK:-500}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
TARGET_ACC="${TARGET_ACC:-1.0}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SKIP_PLOT="${SKIP_PLOT:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"

ARGS=(
  --model-dir "${MODEL_DIR}"
  --model-list "${MODEL_LIST}"
  --models "${MODELS}"
  --train-min-len "${TRAIN_MIN_LEN}"
  --train-max-len "${TRAIN_MAX_LEN}"
  --eval-min-len "${EVAL_MIN_LEN}"
  --eval-max-len "${EVAL_MAX_LEN}"
  --eval-step "${EVAL_STEP}"
  --context-len "${CONTEXT_LEN}"
  --lr "${LR}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-num-batches "${EVAL_NUM_BATCHES}"
  --in-dist-num-batches "${IN_DIST_NUM_BATCHES}"
  --train-steps-per-check "${TRAIN_STEPS_PER_CHECK}"
  --max-train-steps "${MAX_TRAIN_STEPS}"
  --target-acc "${TARGET_ACC}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${SKIP_PLOT}" == "true" ]]; then
  ARGS+=(--skip-plot)
fi

if [[ "${FORCE_TRAIN}" == "true" ]]; then
  ARGS+=(--force-train)
fi

python "${REPO_ROOT}/copying/run_copy_length_generalization.py" "${ARGS[@]}"
