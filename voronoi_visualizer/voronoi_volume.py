"""Monte Carlo Voronoi-volume estimates for decoder embeddings."""

from __future__ import annotations

from math import gamma, pi
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from tqdm.auto import tqdm


DEFAULT_CHUNK_SIZE_TOKENS = 2048
DEFAULT_POINT_BATCH_SIZE = 8192
DEFAULT_CONVEX_HULL_SIMPLEX_SIZE = 64


def _decoder_rows(decoder_matrix: torch.Tensor, tokens_on_columns: bool) -> torch.Tensor:
    matrix = decoder_matrix.detach()
    if tokens_on_columns:
        matrix = matrix.t()
    if matrix.dim() != 2:
        raise ValueError("decoder_matrix must be 2D.")
    return matrix.float()


def _selected_rows(matrix: torch.Tensor, subset_indices: Optional[Sequence[int]]) -> Tuple[torch.Tensor, list[int]]:
    if subset_indices is None:
        token_ids = list(range(matrix.shape[0]))
        return matrix, token_ids
    token_ids = [int(idx) for idx in subset_indices]
    if not token_ids:
        raise ValueError("subset_indices was provided but empty.")
    index = torch.tensor(token_ids, dtype=torch.long)
    return matrix[index], token_ids


def _ball_volume(radius: float, dim: int) -> float:
    return (pi ** (dim / 2.0) / gamma(dim / 2.0 + 1.0)) * (radius ** dim)


def _sample_ball(batch_size: int, dim: int, radius: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    points = torch.randn(batch_size, dim, generator=generator, device=device)
    points = points / points.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scales = torch.rand(batch_size, 1, generator=generator, device=device).pow(1.0 / dim) * radius
    return points * scales


def _sample_rectangle(batch_size: int, bounds: torch.Tensor, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    lows = bounds[:, 0].to(device)
    highs = bounds[:, 1].to(device)
    if torch.any(highs <= lows):
        raise ValueError("Each rectangle bound must satisfy max > min.")
    unit = torch.rand(batch_size, bounds.shape[0], generator=generator, device=device)
    return lows + unit * (highs - lows)


def _sample_convex_hull(
    batch_size: int,
    vertices: torch.Tensor,
    simplex_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if simplex_size <= 0:
        raise ValueError("convex_hull_simplex_size must be positive.")
    num_vertices = vertices.shape[0]
    sample_size = min(simplex_size, num_vertices)
    ids = torch.randint(num_vertices, (batch_size, sample_size), generator=generator, device=device)
    weights = torch.rand(batch_size, sample_size, generator=generator, device=device)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    chosen = vertices[ids]
    return torch.sum(chosen * weights.unsqueeze(-1), dim=1)


def _nearest_token_ids(points: torch.Tensor, token_rows: torch.Tensor, chunk_size_tokens: int) -> torch.Tensor:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")
    best_dist = torch.full((points.shape[0],), float("inf"), device=points.device)
    best_ids = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    point_norm = (points * points).sum(dim=1, keepdim=True)
    for start in range(0, token_rows.shape[0], chunk_size_tokens):
        chunk = token_rows[start:start + chunk_size_tokens]
        token_norm = (chunk * chunk).sum(dim=1).unsqueeze(0)
        dist = point_norm + token_norm - 2.0 * points @ chunk.t()
        chunk_dist, chunk_ids = torch.min(dist, dim=1)
        update = chunk_dist < best_dist
        best_dist[update] = chunk_dist[update]
        best_ids[update] = chunk_ids[update] + start
    return best_ids


def estimate_voronoi_cell_volumes(
    decoder_matrix: torch.Tensor,
    region: str = "ball",
    radius: Optional[float] = None,
    bounds: Optional[torch.Tensor] = None,
    subset_indices: Optional[Sequence[int]] = None,
    num_samples: int = 50000,
    tokens_on_columns: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    point_batch_size: int = DEFAULT_POINT_BATCH_SIZE,
    seed: Optional[int] = None,
    convex_hull_simplex_size: int = DEFAULT_CONVEX_HULL_SIMPLEX_SIZE,
) -> Dict[str, Any]:
    """Estimate decoder-token Voronoi cell proportions by Monte Carlo sampling."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if point_batch_size <= 0:
        raise ValueError("point_batch_size must be positive.")

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = _decoder_rows(decoder_matrix, tokens_on_columns).to(dev)
    token_rows, selected_token_ids = _selected_rows(all_rows, subset_indices)
    dim = int(token_rows.shape[1])

    generator = torch.Generator(device=dev)
    if seed is not None:
        generator.manual_seed(int(seed))

    region = region.lower()
    if region == "ball":
        if radius is None or radius <= 0:
            raise ValueError("radius must be positive when region='ball'.")
        region_volume = _ball_volume(float(radius), dim)
    elif region == "rectangle":
        if bounds is None:
            raise ValueError("bounds must be provided when region='rectangle'.")
        bounds = bounds.float()
        if bounds.shape != (dim, 2):
            raise ValueError(f"bounds must have shape ({dim}, 2).")
        region_volume = float(torch.prod(bounds[:, 1] - bounds[:, 0]).item())
    elif region == "convex_hull":
        region_volume = 1.0
    else:
        raise ValueError("region must be one of: ball, rectangle, convex_hull.")

    counts = torch.zeros(token_rows.shape[0], dtype=torch.long, device=dev)
    total_batches = (num_samples + point_batch_size - 1) // point_batch_size
    progress = tqdm(total=num_samples, desc=f"voronoi:{region}", unit="point")

    for batch_idx in range(total_batches):
        batch_size = min(point_batch_size, num_samples - batch_idx * point_batch_size)
        if region == "ball":
            points = _sample_ball(batch_size, dim, float(radius), generator, dev)
        elif region == "rectangle":
            points = _sample_rectangle(batch_size, bounds, generator, dev)
        else:
            points = _sample_convex_hull(batch_size, token_rows, convex_hull_simplex_size, generator, dev)
        nearest = _nearest_token_ids(points, token_rows, chunk_size_tokens)
        counts += torch.bincount(nearest, minlength=token_rows.shape[0])
        progress.update(batch_size)

    progress.close()
    proportions = counts.float() / float(num_samples)
    volumes = proportions * float(region_volume)

    count_by_token = {tid: int(counts[i].item()) for i, tid in enumerate(selected_token_ids)}
    prop_by_token = {tid: float(proportions[i].item()) for i, tid in enumerate(selected_token_ids)}
    volume_by_token = {tid: float(volumes[i].item()) for i, tid in enumerate(selected_token_ids)}

    return {
        "region": region,
        "radius": radius,
        "dim": dim,
        "num_samples": int(num_samples),
        "region_volume": float(region_volume),
        "selected_token_ids": selected_token_ids,
        "counts": count_by_token,
        "proportion_per_token": prop_by_token,
        "volume_per_token": volume_by_token,
    }
