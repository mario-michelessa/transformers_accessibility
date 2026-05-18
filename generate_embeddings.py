#!/usr/bin/env python3

'''Sample last-token embeddings from random token sequences for one or more HF models.

For each model and each prompt length in PROMPT_LENGTHS, the script draws exactly
MAX_EMBEDDINGS random sequences and saves the resulting embeddings as
`data/embeddings_samples/<model_name>/embedding_{n}.npy`.
'''

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

MODEL_ROOT = Path(os.environ.get('LLM_VIS_MODEL_ROOT', 'models/llms-theory')).expanduser()
EMBEDDING_OUTPUT_DIR = Path('data/embeddings_samples')
MODEL_IDS = [
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1b',
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
    'EleutherAI/pythia-6.9b',
    'Qwen/Qwen2.5-0.5B',
    'Qwen/Qwen2.5-1.5B',
    'meta-llama/Llama-3.2-1B',
    'google/gemma-3-270m',
]
PROMPT_LENGTHS: List[int] = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
MAX_EMBEDDINGS = 10000


def parse_lengths(arg: Optional[str]) -> List[int]:
    if arg is None:
        return list(PROMPT_LENGTHS)
    values: List[int] = []
    for token in arg.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_csv(arg: str) -> List[str]:
    return [piece.strip() for piece in arg.split(',') if piece.strip()]


def resolve_model_path(model_id: str, model_root: Optional[Path], local_files_only: bool) -> str:
    direct = Path(model_id).expanduser()
    if direct.exists():
        return str(direct)
    if model_root is not None:
        candidate = model_root / model_id
        if candidate.exists():
            return str(candidate)
        if local_files_only:
            raise FileNotFoundError(f'Expected local model at {candidate}')
    return model_id


def generate_random_token_sequences(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        generator=generator,
        dtype=torch.long,
    )


def sample_embeddings_for_length(
    model: torch.nn.Module,
    seq_len: int,
    num_sequences: int,
    batch_size: int,
    device: torch.device,
    vocab_size: int,
    hidden_size: int,
    seed: int,
) -> np.ndarray:
    if num_sequences <= 0:
        raise ValueError('num_sequences must be positive')
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + seq_len)

    embeddings = np.empty((num_sequences, hidden_size), dtype=np.float32)
    total_batches = (num_sequences + batch_size - 1) // batch_size
    pbar = tqdm(total=num_sequences, desc=f"len={seq_len}", unit='seq')
    offset = 0
    with torch.no_grad():
        for batch_idx in range(total_batches):
            current = min(batch_size, num_sequences - offset)
            input_ids = generate_random_token_sequences(vocab_size, seq_len, current, device, generator)
            attention_mask = torch.ones_like(input_ids)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            last_hidden = out.last_hidden_state  # [B, T, H]
            last_vecs = last_hidden[:, -1, :]  # fixed length, last position is always valid
            batch_emb = last_vecs.float().cpu().numpy()
            embeddings[offset:offset + current] = batch_emb
            offset += current
            pbar.update(current)
    pbar.close()
    return embeddings


def load_model(model_path: str, device: torch.device, local_files_only: bool) -> torch.nn.Module:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    return model


def model_slug(model_path: str) -> str:
    name = os.path.basename(os.path.normpath(model_path))
    return name if name else model_path.replace('/', '_')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--model-root',
        default=str(MODEL_ROOT),
        help='Directory containing locally downloaded HF models; defaults to $LLM_VIS_MODEL_ROOT or models/llms-theory.',
    )
    p.add_argument(
        '--models',
        default=','.join(MODEL_IDS),
        help='Comma-separated HF model ids or local model directories.',
    )
    p.add_argument('--lengths', default=None, help='Comma-separated prompt lengths (e.g., 8,16,32)')
    p.add_argument(
        '--max-embeddings',
        type=int,
        default=MAX_EMBEDDINGS,
        help='Number of sequences per length (and embeddings saved)',
    )
    p.add_argument('--batch-size', type=int, default=4, help='Inference batch size')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    p.add_argument(
        '--output-dir',
        default=str(EMBEDDING_OUTPUT_DIR),
        help='Base output directory for embedding samples',
    )
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Torch device.')
    p.add_argument('--local-files-only', action='store_true', help='Require models to exist under --model-root.')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    prompt_lengths = parse_lengths(args.lengths)
    model_ids = parse_csv(args.models)
    model_root = Path(args.model_root).expanduser() if args.model_root else None

    os.makedirs(args.output_dir, exist_ok=True)

    for model_id in model_ids:
        model_path = resolve_model_path(model_id, model_root, args.local_files_only)
        print(f"=== Loading model: {model_id} ({model_path}) ===")
        model = load_model(model_path, device, args.local_files_only)
        vocab_size = int(model.get_input_embeddings().weight.shape[0])
        hidden_size = int(getattr(model.config, 'hidden_size', model.get_input_embeddings().weight.shape[1]))
        out_dir = os.path.join(args.output_dir, model_slug(model_id))
        os.makedirs(out_dir, exist_ok=True)

        for seq_len in prompt_lengths:
            if seq_len <= 0:
                raise ValueError(f'Invalid prompt length: {seq_len}')
            print(f'Sampling embeddings: len={seq_len}, count={args.max_embeddings}')
            emb = sample_embeddings_for_length(
                model=model,
                seq_len=seq_len,
                num_sequences=args.max_embeddings,
                batch_size=args.batch_size,
                device=device,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                seed=args.seed,
            )
            out_path = os.path.join(out_dir, f'embedding_{seq_len}.npy')
            np.save(out_path, emb)
            print(f'Saved {emb.shape} to {out_path}')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
