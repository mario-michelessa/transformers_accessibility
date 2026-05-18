"""Artifact and model-list helpers for parent-model copying experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


MODEL_LIST_CANDIDATES = (
    Path("data/models_list"),
    Path("data/models_list.csv"),
    Path("models_list.csv"),
)


def normalize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").lower()


def find_model_list(path_hint: str | None) -> Path:
    candidates: Sequence[Path]
    if path_hint:
        candidates = (Path(path_hint),)
    else:
        candidates = MODEL_LIST_CANDIDATES

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a model list in {candidates}")


def load_model_rows(list_path: Path, subset: Sequence[str] | None) -> List[Tuple[str, str | None]]:
    want = {normalize_name(s) for s in subset} if subset else None
    rows: List[Tuple[str, str | None]] = []

    with open(list_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hf_id = row.get("Link to HF") or row.get("hf") or row.get("HF") or row.get("link")
            if not hf_id:
                values = list(row.values())
                hf_id = values[1] if len(values) >= 2 else None
            if not hf_id:
                continue
            model_name = row.get("Model name") or row.get("Model") or row.get("Name")

            if want:
                candidates = {normalize_name(hf_id), normalize_name(Path(hf_id).name)}
                if model_name:
                    candidates.add(normalize_name(model_name))
                if candidates.isdisjoint(want):
                    continue
            rows.append((hf_id.strip(), model_name.strip() if model_name else None))

    seen = set()
    deduped: List[Tuple[str, str | None]] = []
    for hf_id, model_name in rows:
        key = normalize_name(hf_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((hf_id, model_name))
    return deduped


def model_slug(model_id: str) -> str:
    base = os.path.basename(os.path.normpath(model_id))
    return base if base else model_id.replace("/", "_")


def build_train_signature(args: Any, model_id: str) -> Dict[str, object]:
    return {
        "model": model_id,
        "train_min_len": int(args.train_min_len),
        "train_max_len": int(args.train_max_len),
    }


def signature_id(signature: Dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def cache_dir_for(out_dir: Path, model_tag: str, signature: Dict[str, object]) -> Path:
    return out_dir / "trained_models" / model_tag / signature_id(signature)


def load_cached_model_dir(cache_dir: Path, signature: Dict[str, object]) -> Tuple[bool, int]:
    meta_path = cache_dir / "metadata.json"
    model_dir = cache_dir / "model"
    if not meta_path.exists() or not model_dir.exists():
        return False, 0
    try:
        with open(meta_path) as f_in:
            meta = json.load(f_in)
        if meta.get("signature") != signature:
            return False, 0
        return True, int(meta.get("total_steps", 0))
    except Exception as exc:
        print(f"[cache] Failed to read cached metadata from {cache_dir}: {exc}")
        return False, 0


def save_cached_model(model: Any, cache_dir: Path, signature: Dict[str, object], total_steps: int) -> None:
    model_dir = cache_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    meta_path = cache_dir / "metadata.json"
    with open(meta_path, "w") as f_out:
        json.dump(
            {"signature": signature, "total_steps": int(total_steps)},
            f_out,
            indent=2,
            sort_keys=True,
        )


def write_results(
    out_dir: Path,
    model_tag: str,
    lengths: Sequence[int],
    means: Sequence[float],
    stds: Sequence[float],
    chars: Sequence[float],
    args: Any,
    total_steps: int,
) -> List[Dict[str, object]]:
    model_dir = out_dir / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_path = model_dir / "copy_length.csv"
    log_path = model_dir / "copy_length.log"

    rows: List[Dict[str, object]] = []
    with open(csv_path, "w", newline="") as f_csv, open(log_path, "w") as f_log:
        writer = csv.writer(f_csv)
        writer.writerow([
            "model", "length", "str_acc_mean", "str_acc_std", "char_acc_mean",
            "train_steps", "train_max_len", "context_len", "seed",
        ])
        for length, mean, std, char in zip(lengths, means, stds, chars):
            row = {
                "model": model_tag,
                "length": int(length),
                "str_acc_mean": float(mean),
                "str_acc_std": float(std),
                "char_acc_mean": float(char),
                "train_steps": total_steps,
                "train_max_len": args.train_max_len,
                "context_len": args.context_len,
                "seed": args.seed,
            }
            rows.append(row)
            writer.writerow([row[key] for key in [
                "model", "length", "str_acc_mean", "str_acc_std", "char_acc_mean",
                "train_steps", "train_max_len", "context_len", "seed",
            ]])
            f_log.write(
                f"[copy] model={row['model']} length={row['length']} mean={row['str_acc_mean']:.4f} "
                f"std={row['str_acc_std']:.4f} char={row['char_acc_mean']:.4f} "
                f"steps={row['train_steps']} train_max_len={row['train_max_len']} "
                f"context={row['context_len']} seed={row['seed']}\n"
            )
    return rows


def write_train_progress(out_dir: Path, model_tag: str, rows: List[Dict[str, object]]) -> None:
    model_dir = out_dir / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "train_progress.csv", "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["step", "in_dist_acc"])
        for row in rows:
            writer.writerow([row["step"], row["in_dist_acc"]])


def update_summary(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, object]] = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            existing.extend(csv.DictReader(f))

    dedup: Dict[Tuple[str, int], Dict[str, object]] = {}
    for row in existing + rows:
        dedup[(str(row["model"]), int(row["length"]))] = row

    fieldnames = [
        "model", "length", "str_acc_mean", "str_acc_std", "char_acc_mean",
        "train_steps", "train_max_len", "context_len", "seed",
    ]
    with open(summary_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(dedup.values(), key=lambda x: (x["model"], int(x["length"]))):
            writer.writerow(row)


def plot_model(model_tag: str, rows: List[Dict[str, object]], out_path: Path, train_max_len: int, context_len: int) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable ({exc}); skipping plot for {model_tag}")
        return

    rows_sorted = sorted(rows, key=lambda x: int(x["length"]))
    lengths = [int(r["length"]) for r in rows_sorted]
    means = [float(r["str_acc_mean"]) for r in rows_sorted]
    stds = [float(r["str_acc_std"]) for r in rows_sorted]

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(lengths, means, marker="o", label=model_tag)
    plt.fill_between(lengths, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
    plt.axvline(train_max_len, color="#7b2cbf", linestyle=":", linewidth=2, label="train max len")
    plt.axvline(context_len, color="#2d8f2d", linestyle=":", linewidth=2, label="train context")
    plt.xlabel("Length")
    plt.ylabel("String accuracy")
    plt.title(f"Copy length generalization: {model_tag}")
    plt.xlim(min(lengths), max(max(lengths), context_len))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="svg")
    plt.close()
