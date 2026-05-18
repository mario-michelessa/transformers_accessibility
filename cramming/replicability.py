import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from typing import Optional

# Reuse the core experiment logic from train.py to ensure consistency
from train import run_single_experiment


TEXTS_PATH = Path('data/pg19_valid_1k_chunks.csv')
VOCAB_PATH = Path('data/vocab_100k.txt')
SAVE_DIR = Path('runs')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Estimate perfect replicability vs string length and N_mem tokens')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-160m', help='HF model name')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                        help='Computation dtype')
    parser.add_argument('--use_flash_attention_2', action='store_true', help='Enable flash attention 2')
    parser.add_argument('--N_mem_tokens', type=int, nargs='+', default=[1],
                        help='List of memory token counts to evaluate')
    parser.add_argument('--lengths', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256, 512],
                        help='String/token lengths to evaluate (fixed sweep mode)')
    parser.add_argument('--num_iterations', type=int, default=3000, help='Max optimization steps per sample')
    parser.add_argument('--num_repeats', type=int, default=5, help='Random restarts per string to estimate repeatability')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples per length')
    parser.add_argument('--early_stopping_patience', type=int, default=2000, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for memory optimization')
    parser.add_argument('--beta_1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta_2', type=float, default=0.9, help='AdamW beta2')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--texts_path', type=str, default=str(TEXTS_PATH),
                        help='CSV with a column "text" to sample from')
    parser.add_argument('--vocab_path', type=str, default=str(VOCAB_PATH),
                        help='Vocabulary file used when --shuffled is set')
    parser.add_argument('--shuffled', action='store_true',
                        help='Use random strings from vocab_100k.txt instead of dataset texts')
    parser.add_argument('--save_dir', type=str, default=str(SAVE_DIR), help='Output directory for results/plots')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use, e.g. cuda, cuda:0, or cpu')
    # Adaptive sweep options
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive sweep (auto lengths, high-res near phase change)')
    parser.add_argument('--min_length', type=int, default=100, help='Minimum length for adaptive sweep')
    parser.add_argument('--max_length', type=int, default=2000, help='Maximum length for adaptive sweep')
    # Simplified adaptive sweep controls
    parser.add_argument('--big_step', type=int, default=100, help='Step size while above 0.9')
    parser.add_argument('--small_step', type=int, default=10, help='Dense sampling step between 0.9 and 0.1')
    return parser.parse_args()


def estimate_perfect_replicability(model_name: str,
                                   lengths: list[int],
                                   mem_counts: list[int],
                                   num_samples: int,
                                   num_repeats: int,
                                   num_iterations: int,
                                   dtype: str,
                                   use_flash_attention_2: bool,
                                   early_stopping_patience: int,
                                   lr: float,
                                   beta_1: float,
                                   beta_2: float,
                                   weight_decay: float,
                                   texts: list[str],
                                   shuffled: bool,
                                   vocab_path: str,
                                   device: str = 'cuda',
                                   log_path: Optional[str] = None):
    """
    For each length and N_mem, estimate per-string repeatability via multiple random restarts.
    - For a given string j, run 'num_repeats' independent restarts; let r_j be the fraction of runs
      that reach best_accuracy == 1.0. This is the per-string repeatability.
    - Aggregate across strings to compute:
        * prop_any_success: proportion of strings with r_j > 0
        * avg_repeatability: mean_j r_j
    """
    results = {}
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a list of samples to iterate over
    # - dataset mode: take first num_samples texts
    # - shuffled mode: generate placeholders; text content is ignored downstream
    if shuffled:
        samples = list(range(num_samples))
    else:
        samples = texts[:num_samples]
        print("Lengths of texts tested", [len(sample) for sample in samples])

    def _log(msg: str):
        print(msg, flush=True)
        if log_path is not None:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")

    _log("n_mem,length,done,repeatability,curr_avg_repeatability,best_acc,num_steps")

    for n_mem in mem_counts:
        prop_any_success_list = []
        avg_repeatability_list = []
        details = []
        for L in lengths:
            if n_mem > L:
                # Not meaningful to have more [mem] tokens than the target length
                prop_any_success_list.append(0.0)
                avg_repeatability_list.append(0.0)
                details.append({'length': L, 'num_strings': 0, 'num_strings_any_success': 0, 'sum_repeatability': 0.0})
                continue

            num_strings_any_success = 0
            sum_repeatability = 0.0
            num_str = len(samples)
            for sample_idx, sample in enumerate(samples):
                # For this string, estimate repeatability across random restarts
                successes = 0
                SUCCESS_EPS = 1e-6
                for rep in range(num_repeats):
                    # New random seed to change memory initialization
                    torch.manual_seed(1000003 + L * 1009 + n_mem * 53 + sample_idx * 7 + rep)
                    res = run_single_experiment(
                        N_mem_tokens=n_mem,
                        text_sample=(sample if not shuffled else ""),
                        max_length=L,
                        num_iterations=num_iterations,
                        sample_idx=sample_idx,
                        run_idx=rep,
                        model_name=model_name,
                        dtype=torch_dtype,
                        use_flash_attention_2=use_flash_attention_2,
                        device=device,
                        tokenizer=tokenizer,
                        lr=lr,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        weight_decay=weight_decay,
                        early_stopping_patience=early_stopping_patience,
                        shuffled=shuffled,
                        vocab_path=vocab_path,
                    )
                    if res['best_accuracy'] >= 1.0 - SUCCESS_EPS:
                        successes += 1

                repeatability = successes / max(1, num_repeats)
                sum_repeatability += repeatability
                if successes > 0:
                    num_strings_any_success += 1

                # Logging current averages for this length
                done = sample_idx + 1
                curr_avg_rep = sum_repeatability / max(1, done)
                _log(
                    f"{n_mem},{L},{done},{repeatability:.3f},{curr_avg_rep:.3f},{res['best_accuracy']:.3f},{len(res['losses'])}"
                )

            prop_any_success_list.append(num_strings_any_success / max(1, num_str))
            avg_repeatability_list.append(sum_repeatability / max(1, num_str))
            details.append({
                'length': L,
                'num_strings': num_str,
                'num_strings_any_success': num_strings_any_success,
                'sum_repeatability': sum_repeatability,
            })

        results[n_mem] = {
            'lengths': list(lengths),
            'prop_any_success': prop_any_success_list,
            'avg_repeatability': avg_repeatability_list,
            'details': details,
        }

    return results


def plot_replicability(results: dict, title: str, save_path: Path):
    plt.figure(figsize=(8, 5))
    for n_mem, data in sorted(results.items(), key=lambda x: x[0]):
        plt.plot(data['lengths'], data['prop_any_success'], marker='o', label=f'mem={n_mem}')
    plt.xlabel('String length (tokens)')
    plt.ylabel('Proportion of strings with 100% match')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def sweep_replicability_model(
    model_name: str,
    mem_counts: list[int],
    num_samples: int,
    num_repeats: int,
    num_iterations: int,
    dtype: str,
    use_flash_attention_2: bool,
    early_stopping_patience: int,
    lr: float,
    beta_1: float,
    beta_2: float,
    weight_decay: float,
    texts: list[str],
    shuffled: bool,
    vocab_path: str,
    device: str = 'cuda',
    log_path: Optional[str] = None,
    min_length: int = 100,
    max_length: int = 2000,
    big_step: int = 100,
    small_step: int = 10,
):
    """
    Simplified adaptive sweep per spec:
    - While prop_any_success > 0.9, take big steps.
    - Sample every small_step between 0.9 and 0.1.
    - Once prop_any_success < 0.1, take one extra big step and stop.
    Returns sampled lengths and metrics per N_mem.
    """
    results = {}
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare samples
    if shuffled:
        samples = list(range(num_samples))
    else:
        samples = texts[:num_samples]
        print("Lengths of texts tested", [len(sample) for sample in samples])

    def _log(msg: str):
        print(msg, flush=True)
        if log_path is not None:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")

    _log("n_mem,length,done,repeatability,curr_avg_repeatability,best_acc,num_steps")

    # Helper to evaluate a single length with caching per n_mem
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def evaluate_length(n_mem: int, L: int):
        num_strings_any_success = 0
        sum_repeatability = 0.0
        num_str = len(samples)
        last_res = None
        for sample_idx, sample in enumerate(samples):
            successes = 0
            SUCCESS_EPS = 1e-6
            for rep in range(num_repeats):
                # Seed deterministically per (n_mem, L, sample_idx, rep)
                torch.manual_seed(1000003 + L * 1009 + n_mem * 53 + sample_idx * 7 + rep)
                res = run_single_experiment(
                    N_mem_tokens=n_mem,
                    text_sample=(sample if not shuffled else ""),
                    max_length=L,
                    num_iterations=num_iterations,
                    sample_idx=sample_idx,
                    run_idx=rep,
                    model_name=model_name,
                    dtype=torch_dtype,
                    use_flash_attention_2=use_flash_attention_2,
                    device=device,
                    tokenizer=tokenizer,
                    lr=lr,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    weight_decay=weight_decay,
                    early_stopping_patience=early_stopping_patience,
                    shuffled=shuffled,
                    vocab_path=vocab_path,
                )
                last_res = res
                if res['best_accuracy'] >= 1.0 - SUCCESS_EPS:
                    successes += 1

            repeatability = successes / max(1, num_repeats)
            sum_repeatability += repeatability
            if successes > 0:
                num_strings_any_success += 1

            done = sample_idx + 1
            curr_avg_rep = sum_repeatability / max(1, done)
            _log(
                f"{n_mem},{L},{done},{repeatability:.3f},{curr_avg_rep:.3f},{last_res['best_accuracy']:.3f},{len(last_res['losses'])}"
            )

        prop_any = num_strings_any_success / max(1, num_str)
        avg_rep = sum_repeatability / max(1, num_str)
        detail = {
            'length': L,
            'num_strings': num_str,
            'num_strings_any_success': num_strings_any_success,
            'sum_repeatability': sum_repeatability,
        }
        return prop_any, avg_rep, detail

    for n_mem in mem_counts:
        # Start near min_length but >= n_mem+1
        start_L = max(min_length, n_mem + 1)
        sampled = {}
        details_by_L = {}

        def sample_at(L: int):
            if L < start_L or L > max_length:
                return None
            if L in sampled:
                return sampled[L]
            p_any, avg_rep, det = evaluate_length(n_mem, L)
            sampled[L] = (p_any, avg_rep)
            details_by_L[L] = det
            return sampled[L]

        # 1) Big steps while > 0.9
        L = start_L
        prev_high_L = None
        while L <= max_length:
            vals = sample_at(L)
            if vals is None:
                break
            p_any = vals[0]
            if p_any > 0.9:
                prev_high_L = L
                L += big_step
                continue
            break  # at <= 0.9, move to dense sampling

        # 2) Dense sampling every small_step until < 0.1
        if prev_high_L is not None:
            start_dense = prev_high_L + small_step
        else:
            start_dense = max(L, start_L)

        L_dense = start_dense
        while L_dense <= max_length:
            vals = sample_at(L_dense)
            if vals is None:
                break
            p_any = vals[0]
            if p_any < 0.1:
                # 3) One extra big step and stop
                extra = L_dense + big_step
                if extra <= max_length:
                    sample_at(extra)
                break
            L_dense += small_step

        lengths_sorted = sorted(sampled.keys())
        results[n_mem] = {
            'lengths': lengths_sorted,
            'prop_any_success': [sampled[L][0] for L in lengths_sorted],
            'avg_repeatability': [sampled[L][1] for L in lengths_sorted],
            'details': [details_by_L[L] for L in lengths_sorted],
        }

    return results

def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load texts or set to empty (shuffled mode generates inside run_single_experiment)
    if Path(args.texts_path).exists():
        df = pd.read_csv(args.texts_path, index_col=0)
        texts = df['text'].tolist()
    else:
        texts = []

    device = args.device

    # Prepare output paths early to stream logs
    suffix = 'rnd_vocab_100k' if args.shuffled else 'dataset'
    model_tag = args.model_name.replace('/', '_')
    save_dir = Path(args.save_dir) / model_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(save_dir / f'data_{suffix}.csv')

    if args.adaptive:
        results = sweep_replicability_model(
            model_name=args.model_name,
            mem_counts=args.N_mem_tokens,
            num_samples=args.num_samples,
            num_repeats=args.num_repeats,
            num_iterations=args.num_iterations,
            dtype=args.dtype,
            use_flash_attention_2=args.use_flash_attention_2,
            early_stopping_patience=args.early_stopping_patience,
            lr=args.lr,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            weight_decay=args.weight_decay,
            texts=texts,
            shuffled=args.shuffled,
            vocab_path=args.vocab_path,
            device=device,
            log_path=log_path,
            min_length=args.min_length,
            max_length=args.max_length,
            big_step=args.big_step,
            small_step=args.small_step,
        )
    else:
        results = estimate_perfect_replicability(
            model_name=args.model_name,
            lengths=args.lengths,
            mem_counts=args.N_mem_tokens,
            num_samples=args.num_samples,
            num_repeats=args.num_repeats,
            num_iterations=args.num_iterations,
            dtype=args.dtype,
            use_flash_attention_2=args.use_flash_attention_2,
            early_stopping_patience=args.early_stopping_patience,
            lr=args.lr,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            weight_decay=args.weight_decay,
            texts=texts,
            shuffled=args.shuffled,
            vocab_path=args.vocab_path,
            device=device,
            log_path=log_path,
        )

    # Save plot
    plot_path = save_dir / f'replicability_{suffix}.png'
    title = f'Perfect replicability vs length ({model_tag})'
    plot_replicability(results, title, plot_path)

    # Also save raw results for later analysis
    import json
    with open(save_dir / f'replicability_{suffix}.json', 'w') as f:
        json.dump(results, f)

    print(f'Saved plot to: {plot_path}')
    print(f'Log written to: {log_path}')


if __name__ == '__main__':
    main()
