from __future__ import annotations

from math import exp, log
import math
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


Dist = List[Tuple[float, float]]  # [(value, prob)], sorted by value
VORONOI_RESULTS_DIR = Path("data/voronoi_results")
CONVOLUTION_FIGURE_DIR = Path("figures/convolutions")
CONVOLUTION_DISTRIBUTION_DIR = CONVOLUTION_FIGURE_DIR / "distributions"
MEDIAN_ITERATIONS_PATH = CONVOLUTION_FIGURE_DIR / "median_by_iteration.txt"


def _normalize(d: Dist) -> Dist:
    s = sum(p for _, p in d)
    if s <= 0:
        raise ValueError("Total probability mass must be > 0.")
    out = [(v, p / s) for v, p in d if p > 0]
    out.sort(key=lambda t: t[0])
    return out

def _median(d: Dist) -> float:
    """d must be sorted by value and normalized."""
    c = 0.0
    for v, p in d:
        c += p
        if c >= 0.5:
            return v
    return d[-1][0]  # numerical fallback

def _compress_to_n(d: Dist, n: int, use_log: bool = False) -> Dist:
    """
    Compress a sorted discrete distribution to at most n support points
    by grouping consecutive points into ~equal-probability bins.

    Representative value per bin:
      - weighted geometric mean if all values in bin are > 0
      - otherwise weighted arithmetic mean
    """
    if n <= 0:
        raise ValueError("n must be >= 1.")
    if len(d) <= n:
        return d
    values = np.array([v for v, _ in d], dtype=float)
    probs = np.array([p for _, p in d], dtype=float)
    return _compress_sorted_arrays_to_n(values, probs, n, use_log=use_log)


def _mul_convolve(dk: Dist, d: Dist) -> Dist:
    """Exact (dense) multiplication-convolution: support of dk * support of d."""
    acc: Dict[float, float] = {}
    for x, px in dk:
        for y, py in d:
            v = x * y
            acc[v] = acc.get(v, 0.0) + px * py
    out = [(v, p) for v, p in acc.items() if p > 0]
    out.sort(key=lambda t: t[0])
    return out


def _add_convolve(dk: Dist, d: Dist) -> Dist:
    """Exact (dense) additive convolution: support of dk + support of d."""
    acc: Dict[float, float] = {}
    for x, px in dk:
        for y, py in d:
            v = x + y
            acc[v] = acc.get(v, 0.0) + px * py
    out = [(v, p) for v, p in acc.items() if p > 0]
    out.sort(key=lambda t: t[0])
    return out


def f(n: int, d: Dist, epsilon: float, max_k: int = 10000) -> int:
    """
    Inputs
      - n: max support size used to represent each intermediate distribution (compression budget)
      - d: base discrete distribution of X (ordered list of [value, prob])
      - epsilon: stopping threshold

    Procedure
      For k = 1, 2, ...
        compute d_k = distribution of product X_1 * ... * X_k (iid X_i ~ d),
        represent d_k with at most n points (compression),
        stop when median(d_k) < epsilon and return k.

    Notes
      - This computes the product distribution exactly at each step, then compresses to <= n points.
      - If the stopping condition never happens within max_k, raises RuntimeError.
    """
    if epsilon != epsilon:
        raise ValueError("epsilon must be a real number (not NaN).")
    if max_k < 1:
        raise ValueError("max_k must be >= 1.")

    d0 = _normalize([(float(v), float(p)) for v, p in d])
    dk = d0  # k=1 distribution

    # k = 1 check
    dk = _compress_to_n(dk, n)
    if _median(dk) < epsilon:
        return 1

    for k in range(2, max_k + 1):
        dk = _mul_convolve(dk, d0)
        dk = _normalize(dk)
        dk = _compress_to_n(dk, n)
        if _median(dk) < epsilon:
            return k

    raise RuntimeError(f"Stopping condition not met for k <= {max_k}.")




# Usage example:
# d = [(0.5, 0.6), (2.0, 0.4)]   # X in {0.5, 2} with given probs
# k = f(n=200, d=d, epsilon=1e-3)
# print(k)

def _compress_sorted_arrays_to_n(
    values: np.ndarray,
    probs: np.ndarray,
    n: int,
    use_log: bool = False,
) -> Dist:
    if n <= 0:
        raise ValueError("n must be >= 1.")
    if values.size <= n:
        return [(float(v), float(p)) for v, p in zip(values, probs)]
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("Total probability mass must be > 0.")
    probs = probs / total

    cdf = np.cumsum(probs)
    targets = np.linspace(1.0 / n, 1.0, n)
    ends = np.searchsorted(cdf, targets, side="left")
    ends = np.clip(ends, 0, values.size - 1)
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1

    csum_p = np.cumsum(probs)
    csum_v = np.cumsum(probs * values)

    def _range_sum(csum):
        out = csum[ends]
        out = out - np.where(starts > 0, csum[starts - 1], 0.0)
        return np.where(starts <= ends, out, 0.0)

    mass = _range_sum(csum_p)
    weighted = _range_sum(csum_v)

    if use_log:
        reps = np.divide(weighted, mass, out=np.zeros_like(weighted), where=mass > 0)
    else:
        if np.all(values > 0):
            csum_log = np.cumsum(probs * np.log(values))
            weighted_log = _range_sum(csum_log)
            reps = np.exp(np.divide(weighted_log, mass, out=np.zeros_like(weighted_log), where=mass > 0))
        else:
            reps = np.divide(weighted, mass, out=np.zeros_like(weighted), where=mass > 0)

    mask = mass > 0
    return [(float(v), float(p)) for v, p in zip(reps[mask], mass[mask])]


def _convolve_arrays(dk: Dist, d: Dist, log_values: bool) -> Tuple[np.ndarray, np.ndarray]:
    v1 = np.array([v for v, _ in dk], dtype=float)
    p1 = np.array([p for _, p in dk], dtype=float)
    v2 = np.array([v for v, _ in d], dtype=float)
    p2 = np.array([p for _, p in d], dtype=float)
    if log_values:
        values = v1[:, None] + v2[None, :]
    else:
        values = v1[:, None] * v2[None, :]
    probs = p1[:, None] * p2[None, :]
    return values.ravel(), probs.ravel()


def _convolve_and_compress(
    dk: Dist,
    d: Dist,
    n: int,
    log_values: bool,
    use_log: bool,
) -> Dist:
    values, probs = _convolve_arrays(dk, d, log_values=log_values)
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("Total probability mass must be > 0.")
    probs = probs / total
    order = np.argsort(values)
    values = values[order]
    probs = probs[order]
    return _compress_sorted_arrays_to_n(values, probs, n, use_log=use_log)


def _plot_overlaid_hist(
    ax,
    dks: Dict[int, Dist],
    num_bins: int,
    log_values: bool,
    title: Optional[str] = None,
) -> bool:
    all_vals = np.concatenate([np.array([v for v, _ in dks[k]], dtype=float) for k in dks])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return False

    if log_values:
        log10_vals = all_vals / log(10.0)
        log_min = log10_vals.min()
        log_max = log10_vals.max()
    else:
        pos = all_vals[all_vals > 0]
        if pos.size == 0:
            ax.text(0.5, 0.5, "no positive data", ha="center", va="center")
            ax.set_axis_off()
            return False
        log_min = np.log10(pos.min())
        log_max = np.log10(pos.max())

    bins = np.linspace(log_min, log_max, num_bins + 1)

    for k in sorted(dks):
        vals = np.array([v for v, _ in dks[k]], dtype=float)
        weights = np.array([p for _, p in dks[k]], dtype=float)
        mask = np.isfinite(vals) & np.isfinite(weights)
        if log_values:
            log10 = vals[mask] / log(10.0)
        else:
            mask = mask & (vals > 0)
            log10 = np.log10(vals[mask])
        hist, edges = np.histogram(log10, bins=bins, weights=weights[mask])
        if hist.sum() > 0:
            hist = hist / hist.sum()
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.step(10 ** centers, hist, where="mid", label=f"k={k}", alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("volume (proportion)")
    ax.set_ylabel("prob mass")
    ax.grid(alpha=0.2, which="both")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return True

def plot_convolution_histogram_single(
    model_name: str,
    dks: Dict[int, Dist],
    out_path: Path,
    num_bins: int = 70,
    log_values: bool = False,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    _plot_overlaid_hist(
        ax,
        dks=dks,
        num_bins=num_bins,
        log_values=log_values,
        title=model_name.split("/")[-1],
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def _dist_from_proportions(props, weight_mode="uniform", log_values: bool = False):
    props = np.asarray(props, dtype=float)
    props = props[props > 0]
    if props.size == 0:
        raise ValueError("No positive proportions.")
    if weight_mode == "uniform":
        weights = np.ones_like(props, dtype=float)
    elif weight_mode == "volume":
        weights = props.astype(float)
    else:
        raise ValueError("weight_mode must be 'uniform' or 'volume'")
    values = np.log(props) if log_values else props
    d = [(float(v), float(w)) for v, w in zip(values, weights)]
    return _normalize(d), int(props.size)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def compute_median_trajectories(
    models,
    csv_dir: Path,
    epsilon: float,
    max_k: int = 12,
    n: int = 2000,
    weight_mode: str = "uniform",
    log_values: bool = False,
    conv_method: str = "numpy",
    verbose: bool = True,
    dist_plot_dir: Optional[Path] = None,
    dist_num_bins: int = 70,
    out_txt: Optional[Path] = None,
):
    csv_dir = Path(csv_dir)
    if out_txt is None:
        out_txt = MEDIAN_ITERATIONS_PATH
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[Tuple[int, float]]] = {}

    if log_values and epsilon <= 0:
        raise ValueError("epsilon must be > 0 when log_values is True.")
    log_epsilon = log(epsilon) if log_values else None

    if conv_method not in {"numpy", "python"}:
        raise ValueError("conv_method must be 'numpy' or 'python'")

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("model,iteration,median\n")
        f.flush()

        total_models = len(models)
        for model_idx, model_name in enumerate(models, start=1):
            if verbose:
                bar = _progress_bar(model_idx, total_models)
                print(f"{bar} model {model_idx}/{total_models}: {model_name}")
            csv_path = csv_dir / f"{_safe_model_name(model_name)}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing Voronoi result CSV: {csv_path}")

            df = pd.read_csv(csv_path)
            if "proportion" not in df.columns:
                raise ValueError(f"Missing 'proportion' column in {csv_path}")

            d0, _ = _dist_from_proportions(
                df["proportion"].values,
                weight_mode=weight_mode,
                log_values=log_values,
            )
            dk = _compress_to_n(d0, n, use_log=log_values)
            dks = {1: dk} if dist_plot_dir is not None else None

            med_raw = _median(dk)
            med = exp(med_raw) if log_values else med_raw
            results[model_name] = [(1, med)]
            f.write(f"{model_name},1,{med:.10g}\n")
            f.flush()

            stop = (med_raw < log_epsilon) if log_values else (med < epsilon)
            if verbose:
                iter_bar = _progress_bar(1, max_k)
                print(f"  {iter_bar} iter 1/{max_k} median={med:.6g}")
            if stop:
                continue

            for k in range(2, max_k + 1):
                if conv_method == "numpy":
                    dk = _convolve_and_compress(
                        dk,
                        d0,
                        n=n,
                        log_values=log_values,
                        use_log=log_values,
                    )
                else:
                    dk = _add_convolve(dk, d0) if log_values else _mul_convolve(dk, d0)
                    dk = _normalize(dk)
                    dk = _compress_to_n(dk, n, use_log=log_values)
                if dks is not None:
                    dks[k] = dk
                med_raw = _median(dk)
                med = exp(med_raw) if log_values else med_raw
                results[model_name].append((k, med))
                f.write(f"{model_name},{k},{med:.10g}\n")
                f.flush()
                stop = (med_raw < log_epsilon) if log_values else (med < epsilon)
                if verbose:
                    iter_bar = _progress_bar(k, max_k)
                    print(f"  {iter_bar} iter {k}/{max_k} median={med:.6g}")
                if stop:
                    break

            if dks is not None:
                out_dir = Path(dist_plot_dir)
                out_path = out_dir / f"{_safe_model_name(model_name)}_distributions.pdf"
                plot_convolution_histogram_single(
                    model_name,
                    dks=dks,
                    out_path=out_path,
                    num_bins=dist_num_bins,
                    log_values=log_values,
                )

    return results


def plot_median_decay(
    median_by_model: Dict[str, List[Tuple[int, float]]],
    out_dir: Path,
    log_x: bool = False,
    log_y: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, series in median_by_model.items():
        if not series:
            continue
        ks = [k for k, _ in series]
        meds = [m for _, m in series]
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        ax.plot(ks, meds, marker="o", linewidth=1.6)
        ax.set_xlabel("iteration")
        ax.set_ylabel("median volume")
        ax.set_title(model_name)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"{_safe_model_name(model_name)}_median_decay.pdf"
        fig.savefig(out_path)
        plt.close(fig)

# Assumes these exist from your code:
# _normalize, _compress_to_n, _mul_convolve, _dist_from_proportions

def plot_convolution_histograms_grid(
    models,
    csv_dir: Path,
    max_k: int = 4,
    n: int = 2000,
    weight_mode: str = "uniform",
    num_bins: int = 70,
    log_values: bool = False,
    conv_method: str = "numpy",
):
    cols = 3
    rows = math.ceil(len(models) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.4))
    axes = np.array(axes).reshape(-1)

    if conv_method not in {"numpy", "python"}:
        raise ValueError("conv_method must be 'numpy' or 'python'")

    for ax, model_name in zip(axes, models):
        csv_path = csv_dir / f"{_safe_model_name(model_name)}.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, "missing CSV", ha="center", va="center")
            ax.set_axis_off()
            continue

        df = pd.read_csv(csv_path)
        d0, _ = _dist_from_proportions(
            df["proportion"].values,
            weight_mode=weight_mode,
            log_values=log_values,
        )

        dks = {}
        dk = _compress_to_n(d0, n, use_log=log_values)
        dks[1] = dk
        for k in range(2, max_k + 1):
            if conv_method == "numpy":
                dk = _convolve_and_compress(
                    dk,
                    d0,
                    n=n,
                    log_values=log_values,
                    use_log=log_values,
                )
            else:
                dk = _add_convolve(dk, d0) if log_values else _mul_convolve(dk, d0)
                dk = _normalize(dk)
                dk = _compress_to_n(dk, n, use_log=log_values)
            dks[k] = dk

        _plot_overlaid_hist(
            ax,
            dks=dks,
            num_bins=num_bins,
            log_values=log_values,
            title=model_name.split("/")[-1],
        )

    for ax in axes[len(models):]:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    models = [
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "meta-llama/Llama-3.2-1B",
        "google/gemma-3-270m",
    ]

    medians = compute_median_trajectories(
        models=models,
        csv_dir=VORONOI_RESULTS_DIR,
        epsilon=1e-100,
        max_k=6,
        n=2000,
        weight_mode="uniform",
        log_values=True,
        dist_plot_dir=CONVOLUTION_DISTRIBUTION_DIR,
        dist_num_bins=100,
    )
    plot_median_decay(
        medians,
        out_dir=CONVOLUTION_FIGURE_DIR,
        log_x=False,
        log_y=True,
    )
