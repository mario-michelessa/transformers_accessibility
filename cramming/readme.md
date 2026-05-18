# Exploring Language Models Embeddings Space Capacity

This repository contains code and notebooks for cramming experiments, adapted from : 
Kuratov, Y., Arkhipov, M., Bulatov, A., Burtsev, M., "Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity", ACL 2025.


## Scripts
- `scripts/run_replicability_adaptive.sh` - bash script for estimating accessibility of different models, given `[mem]` vectors.

## Visualizations
- `plots_accessibility.ipynb` - Code for paper figures per model.
- `figures/` - figures per model.

## Data
### Downloading Preprocessed Data

To quickly get started, you can download our preprocessed text chunks for PG-19 and fanfics with a single command:

```bash
cd ./data
./download_texts.sh
```

This script will fetch the required texts and place them in the `./data` folder.
