#!/usr/bin/env python3
"""
Simple script to download a hard-coded list of models
from Pythia, Qwen, and Llama suites and save_pretrained
them into a single output folder on this server.

Edit MODEL_OUTPUT_DIR and MODEL_IDS below, or override them from the CLI.

Auth: For gated models (e.g., Llama), set env HF_TOKEN.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List


MODEL_OUTPUT_DIR = Path(os.environ.get("LLM_VIS_MODEL_ROOT", "models/llms-theory")).expanduser()

MODEL_IDS: List[str] = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-3-1b-pt",
    "google/gemma-3-270m"
]

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_one(model_id: str, out_dir: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dst = out_dir / model_id.replace("/", os.sep)
    ensure_dir(dst)

    common_kwargs = {"trust_remote_code": True}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        common_kwargs["token"] = hf_token

    log(f"Loading: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **common_kwargs,
    )
    tok = AutoTokenizer.from_pretrained(model_id, **common_kwargs)

    log(f"Saving to: {dst}")
    model.save_pretrained(dst)
    tok.save_pretrained(dst)
    log(f"Done: {model_id}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(MODEL_OUTPUT_DIR),
        help="Destination directory for save_pretrained outputs.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_IDS),
        help="Comma-separated Hugging Face model ids to download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    model_ids = [piece.strip() for piece in args.models.split(",") if piece.strip()]

    ensure_dir(out_dir)

    for mid in model_ids:
        save_one(mid, out_dir)

    log("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
