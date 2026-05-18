#!/usr/bin/env python3
"""Fine-tune parent HF models on synthetic copy and evaluate length generalization."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

try:
    from .data_utils import get_tokenizer, get_train_dataset
    from .parent_copy_artifacts import (
        build_train_signature,
        cache_dir_for,
        find_model_list,
        load_cached_model_dir,
        load_model_rows,
        model_slug,
        plot_model,
        save_cached_model,
        update_summary,
        write_results,
        write_train_progress,
    )
    from .test_utils import evaluation
    from .train_utils import ce_loss, custom_get_scheduler, get_optimizer
except ImportError:
    from data_utils import get_tokenizer, get_train_dataset
    from parent_copy_artifacts import (
        build_train_signature,
        cache_dir_for,
        find_model_list,
        load_cached_model_dir,
        load_model_rows,
        model_slug,
        plot_model,
        save_cached_model,
        update_summary,
        write_results,
        write_train_progress,
    )
    from test_utils import evaluation
    from train_utils import ce_loss, custom_get_scheduler, get_optimizer

MODEL_ROOT = os.environ.get("LLM_VIS_MODEL_ROOT", "")
OUTPUT_DIR = Path("copying/synthetic_parent_results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-list",
        default=None,
        help='Path to csv with a "Link to HF" column. Defaults to data/models_list or models_list.csv.',
    )
    p.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated subset of models to run (matches hf id, base name, or model name column).",
    )
    p.add_argument("--model-dir", default=MODEL_ROOT, help="Base directory where local models are stored.")
    p.add_argument("--train-min-len", type=int, default=5, help="Minimum training length.")
    p.add_argument("--train-max-len", type=int, default=50, help="Maximum training length.")
    p.add_argument("--eval-min-len", type=int, default=10, help="Minimum evaluation length.")
    p.add_argument("--eval-max-len", type=int, default=200, help="Maximum evaluation length.")
    p.add_argument("--eval-step", type=int, default=5, help="Step size between eval lengths.")
    p.add_argument(
        "--eval-lengths",
        default=None,
        help="Optional comma-separated explicit evaluation lengths. Overrides eval-min/eval-max/eval-step.",
    )
    p.add_argument("--context-len", type=int, default=512, help="Training/eval context length.")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    p.add_argument("--train-batch-size", type=int, default=8, help="Training batch size.")
    p.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size.")
    p.add_argument("--eval-num-batches", type=int, default=3, help="Number of eval batches.")
    p.add_argument("--in-dist-num-batches", type=int, default=3, help="Eval batches for in-distribution check.")
    p.add_argument("--train-steps-per-check", type=int, default=500, help="Training steps between checks.")
    p.add_argument("--max-train-steps", type=int, default=10000, help="Max training steps.")
    p.add_argument(
        "--target-acc",
        type=float,
        default=1.0,
        help="Target in-distribution string accuracy before stopping.",
    )
    p.add_argument("--device", default="cuda", help="Device to train on.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory.",
    )
    p.add_argument(
        "--force-train",
        action="store_true",
        help="Ignore cached checkpoints and retrain models from scratch.",
    )
    p.add_argument("--skip-plot", action="store_true", help="Skip plotting SVGs.")
    return p.parse_args()


def resolve_eval_lengths(args: argparse.Namespace) -> List[int]:
    if args.eval_lengths:
        values = [int(piece.strip()) for piece in args.eval_lengths.split(",") if piece.strip()]
        if not values:
            raise ValueError("--eval-lengths was provided but no valid integers were found.")
        return values
    return list(range(args.eval_min_len, args.eval_max_len + 1, max(args.eval_step, 1)))


def resolve_model_path(model_id: str, base_dir: str | Path | None) -> str:
    if base_dir:
        candidate = Path(base_dir) / model_id
        if candidate.exists():
            return str(candidate)
    return model_id


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(base: argparse.Namespace, model_name: str) -> SimpleNamespace:
    eval_lengths = resolve_eval_lengths(base)
    return SimpleNamespace(
        train_task="copy",
        eval_task="copy",
        vocab_size=26,
        n_gram=0,
        length_answer=0,
        model=model_name,
        lr=base.lr,
        epochs=1,
        steps=base.train_steps_per_check,
        train_batch_size=base.train_batch_size,
        eval_batch_size=base.eval_batch_size,
        eval_num_batches=base.eval_num_batches,
        min_train_len=base.train_min_len,
        max_train_len=base.train_max_len,
        min_eval_len=min(eval_lengths),
        max_eval_len=max(eval_lengths),
        context_len=base.context_len,
        device=base.device,
    )


def train_for_steps(
    model: torch.nn.Module,
    train_dataset,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: SimpleNamespace,
    to_token: Dict[str, int],
    device: torch.device,
    steps: int,
    start_step: int,
) -> int:
    model.train()
    for idx, batch in enumerate(train_dataset, start=1):
        if idx > steps:
            break
        x = batch["input_ids"][:, :-1].to(device)
        y = batch["input_ids"][:, 1:].to(device)
        mask = batch["mask"][:, 1:].to(device)

        outputs = model(x)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = ce_loss(y, logits, mask, to_token)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return start_step + min(idx, steps)


def evaluate_lengths(
    args: SimpleNamespace,
    model: torch.nn.Module,
    tokenizer,
    to_token: Dict[str, int],
    lengths: Sequence[int],
    num_batches: int,
) -> Tuple[List[float], List[float], List[float]]:
    eval_args = SimpleNamespace(**vars(args))
    eval_args.eval_task = "copy"
    eval_args.eval_lengths = list(lengths)
    eval_args.eval_num_batches = num_batches
    eval_args.quiet = True
    means, stds, chars = evaluation(eval_args, model, tokenizer, to_token)
    return means, stds, chars


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    eval_lengths = resolve_eval_lengths(args)

    if args.context_len <= 2 * args.train_max_len:
        raise ValueError(
            f"context_len ({args.context_len}) must exceed 2 * train_max_len ({args.train_max_len}) "
            "so training strings fit the context."
        )
    if args.context_len <= 2 * max(eval_lengths):
        raise ValueError(
            f"context_len ({args.context_len}) must exceed 2 * max(eval_lengths) ({max(eval_lengths)}) "
            "so evaluation strings fit the context."
        )

    list_path = find_model_list(args.model_list)
    subset = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    model_rows = load_model_rows(list_path, subset)
    if not model_rows:
        raise RuntimeError("No models matched the provided filters.")

    print(f"Using model list: {list_path}")
    out_dir = Path(args.output_dir)
    all_rows: List[Dict[str, object]] = []

    for hf_id, _ in model_rows:
        model_tag = model_slug(hf_id)
        resolved_path = resolve_model_path(hf_id, args.model_dir)
        print(f"\n=== Training model: {hf_id} (resolved: {resolved_path}) ===")

        run_args = build_args(args, hf_id)
        tokenizer, to_token, _ = get_tokenizer(run_args)
        signature = build_train_signature(args, hf_id)
        cache_dir = cache_dir_for(out_dir, model_tag, signature)

        total_steps = 0
        cached, total_steps = (False, 0)
        model_path = resolved_path
        if not args.force_train:
            cached, total_steps = load_cached_model_dir(cache_dir, signature)
            if cached:
                model_path = str(cache_dir / "model")
                print(f"[cache] Loaded trained model from {cache_dir}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        device = torch.device(args.device)
        model.to(device)

        if not cached:
            train_dataset = get_train_dataset(run_args, tokenizer)
            optimizer = get_optimizer(model, run_args)
            scheduler = custom_get_scheduler(optimizer, args.max_train_steps)

            in_dist_len = args.train_max_len
            progress_rows: List[Dict[str, object]] = []
            while total_steps < args.max_train_steps:
                total_steps = train_for_steps(
                    model,
                    train_dataset,
                    optimizer,
                    scheduler,
                    run_args,
                    to_token,
                    device,
                    steps=args.train_steps_per_check,
                    start_step=total_steps,
                )
                means, _, _ = evaluate_lengths(
                    run_args,
                    model,
                    tokenizer,
                    to_token,
                    lengths=[in_dist_len],
                    num_batches=args.in_dist_num_batches,
                )
                in_dist_acc = float(means[0])
                progress_rows.append({"step": total_steps, "in_dist_acc": in_dist_acc})
                print(f"[check] step={total_steps} in_dist_len={in_dist_len} acc={in_dist_acc:.4f}")
                if in_dist_acc >= args.target_acc:
                    print("[check] target reached.")
                    break

            write_train_progress(out_dir, model_tag, progress_rows)
            save_cached_model(model, cache_dir, signature, total_steps)

        means, stds, chars = evaluate_lengths(
            run_args,
            model,
            tokenizer,
            to_token,
            lengths=eval_lengths,
            num_batches=args.eval_num_batches,
        )

        rows = write_results(out_dir, model_tag, eval_lengths, means, stds, chars, args, total_steps)
        all_rows.extend(rows)

        if not args.skip_plot:
            plot_path = out_dir / "plots" / f"{model_tag}.svg"
            plot_model(model_tag, rows, plot_path, args.train_max_len, args.context_len)
            print(f"[plot] saved {plot_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = out_dir / "summary.csv"
    update_summary(summary_path, all_rows)
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
