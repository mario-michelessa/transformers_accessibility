#!/usr/bin/env python3
"""
Plane Visualization Example - Token Prediction Regions on PCA Plane

This script demonstrates the new plane-based visualization approach:
1. Takes three input texts
2. Extracts their embeddings 
3. Uses PCA to define a 2D plane through the embeddings
4. For each pixel on the plane, determines which token would be predicted next
5. Colors each token region with a unique color (max 50 colors, rest gray)
6. Shows the exact convex regions where each token is most likely
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.environ.get("LLM_VIS_VORONOI_MODEL", "Qwen/Qwen2.5-0.5B")
OUTPUT_PATH = Path("voronoi_visualizer/qwen_token_plane_visualization.svg")
RESOLUTION = 200
MAX_COLORS = 50
PLANE_RADIUS_MULTIPLIER = 1.1
EXAMPLE_TEXTS = [
    "The quick brown fox",
    "https://",
    "In a distant future",
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
from voronoi_visualizer.model_factory import load_model
from voronoi_visualizer.token_partition_analyzer import TokenPartitionAnalyzer
from voronoi_visualizer.embedding_space_visualizer import EmbeddingSpaceVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME, help="HF model id or local model directory.")
    parser.add_argument("--output-path", default=str(OUTPUT_PATH), help="SVG output path.")
    parser.add_argument("--resolution", type=int, default=RESOLUTION, help="Plane grid resolution.")
    parser.add_argument("--max-colors", type=int, default=MAX_COLORS, help="Number of token regions to color.")
    parser.add_argument(
        "--plane-radius-multiplier",
        type=float,
        default=PLANE_RADIUS_MULTIPLIER,
        help="Multiplier for the PCA plane bounds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_model(args.model_name)
    print("Model loaded successfully.")

    vocab_size = len(model.tokenizer)
    embedding_dim = model.get_output_projection_matrix().shape[1]
    print(f"Vocabulary: {vocab_size:,} tokens")
    print(f"Embedding dimension: {embedding_dim}")
    
    analyzer = TokenPartitionAnalyzer(model)
    visualizer = EmbeddingSpaceVisualizer(analyzer)

    print(f"Using {len(EXAMPLE_TEXTS)} texts to define PCA plane:")
    for i, text in enumerate(EXAMPLE_TEXTS, 1):
        print(f"   {i}. '{text}'")

    print(f"Computing text embeddings for {model.get_num_layers()} layers...")
    text_embeddings = [model.extract_last_token_embedding(text) for text in EXAMPLE_TEXTS]
    
    print("Creating token prediction plane visualization...")
    text_embeddings = torch.stack(text_embeddings).squeeze(1)  # Shape: (3, embedding_dim)
    fig = visualizer.plot_token_regions_from_texts(
        texts=EXAMPLE_TEXTS,
        text_embeddings=text_embeddings,
        resolution=args.resolution,
        max_colors=args.max_colors,
        plane_radius_multiplier=args.plane_radius_multiplier,
    )   

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', format='svg')
    print(f"Visualization saved: {output_path}")
    plt.close(fig)
    return 0
    

if __name__ == "__main__":
    raise SystemExit(main())
    
