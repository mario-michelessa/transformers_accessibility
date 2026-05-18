#!/usr/bin/env bash
set -euo pipefail

DATASETS=("pg19_valid_1k_chunks" "fanfics_1k_chunks")
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for DATASET in "${DATASETS[@]}"; do
    echo "Downloading $DATASET"
    python -c "import datasets; datasets.load_dataset('yurakuratov/${DATASET}', split='train').to_csv('${OUTPUT_DIR}/${DATASET}.csv', index=False)"
done
