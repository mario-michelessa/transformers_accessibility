#!/usr/bin/env bash
set -euo pipefail

# Sample last-token embeddings used by the support/theorem-bound notebook.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${LLM_VIS_MODEL_ROOT:-models/llms-theory}"
OUTPUT_DIR="${REPO_ROOT}/data/embeddings_samples"
MODELS=${1:-"EleutherAI/pythia-160m,EleutherAI/pythia-410m,EleutherAI/pythia-1b,EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,meta-llama/Llama-3.2-1B,google/gemma-3-270m"}
LENGTHS=${2:-"4,8,16,32,64,128,256,512,1024,2048,4096"}
MAX_EMBEDDINGS=${MAX_EMBEDDINGS:-10000}
BATCH_SIZE=${BATCH_SIZE:-4}
DEVICE=${DEVICE:-cuda}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-false}

echo "Models:  ${MODELS}"
echo "Lengths: ${LENGTHS}"
echo "Output:  ${OUTPUT_DIR}"

ARGS=(
  --model-root "${MODEL_ROOT}"
  --models "${MODELS}"
  --lengths "${LENGTHS}"
  --max-embeddings "${MAX_EMBEDDINGS}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${LOCAL_FILES_ONLY}" == "true" ]]; then
  ARGS+=(--local-files-only)
fi

python "${REPO_ROOT}/generate_embeddings.py" "${ARGS[@]}"
