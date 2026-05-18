# "How Many Different Outputs can a Tranformer Generate?": Replication Code

Accepted at ICML 2026 (Spotlight).

## Setup

Use Python 3.9+ and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional model cache location:

```bash
export LLM_VIS_MODEL_ROOT="$PWD/models/llms-theory"
```

For gated Hugging Face models, also set `HF_TOKEN`.

To download the default model set into `LLM_VIS_MODEL_ROOT`:

```bash
python download_llms.py --output-dir "$LLM_VIS_MODEL_ROOT"
```

## Data

Download the cramming text chunks:

```bash
bash cramming/data/download_texts.sh
```

If you use `--shuffled` in cramming, create the vocabulary file:

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python cramming/make_vocab.py --glove_path glove.6B/glove.6B.50d.txt --vocab_size 100000 --output_path cramming/data/vocab_100k.txt
```

## Voronoi Cell Visualization

Generate the PCA-plane next-token Voronoi visualization:

```bash
python -m voronoi_visualizer.simple_plane_visualization_example \
  --model-name Qwen/Qwen2.5-0.5B \
  --output-path figures/qwen_token_plane_visualization.svg \
  --resolution 200
```

If using local models, set `LLM_VIS_MODEL_ROOT` first.

## Cramming Experiments

Run the adaptive accessibility sweep:

```bash
bash cramming/scripts/run_replicability_adaptive.sh
```

The editable model list, memory-token counts, sample counts, and output paths are at the top of that shell script. Outputs are written to `cramming/runs/`.

## Embedding Sampling for Support Estimation

Sample last-token embeddings:

```bash
bash scripts/run_embedding_geometry.sh
```

Outputs are saved under `data/embeddings_samples/<model>/embedding_<length>.npy`.

## Theorem Bounds

After generating embeddings, open and run `th_bounds_estimation.ipynb`.

The notebook uses `data/embeddings_samples`, writes rectangle bounds to `data/rects_inf`, and reads convolution summaries from `figures/convolutions/median_by_iteration.txt`.

To generate convolution summaries from Voronoi result CSVs:

```bash
python cell_convolution.py
```

## Copying Task

Run copy length generalization:

```bash
bash scripts/run_copy_length_generalization.sh
```

Edit `scripts/run_copy_length_generalization.sh` to set the model list, model directory, device, training lengths, evaluation lengths, and output directory. Outputs go to `copying/synthetic_parent_results/` by default.

## Special Thanks 

Codes were adapted from the following repositories: 
- Cramming: [Kuratov et al., 2025](https://github.com/yurakuratov/hidden_capacity)
- Copying: [Jelassi et al., 2024](https://github.com/sjelassi/transformers_ssm_copy)

## Citation 

```bibtex
@inproceedings{
meyer2026how,
title={How Many Different Outputs can a Transformer Generate?},
author={Meyer M., Michelessa M., Chaux C., Tan V.},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={}
}
```
