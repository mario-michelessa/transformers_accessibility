from typing import List, Dict, Any, Optional, Tuple, Set
import torch
import numpy as np
from dataclasses import dataclass
from .base_llm import BaseLLM
from .voronoi_volume import estimate_voronoi_cell_volumes


@dataclass
class TokenRegion:
    """Represents a convex region in embedding space where a token is most likely."""
    token_id: int
    token_str: str
    constraints: List[Tuple[torch.Tensor, float]]  # List of (normal_vector, bias) for Ax <= b
    center: Optional[torch.Tensor] = None  # Representative point in the region
    volume_estimate: Optional[float] = None


class TokenPartitionAnalyzer:
    """Analyzes the embedding space partitions for next token prediction."""
    
    def __init__(self, model: BaseLLM):
        self.model = model
        self.output_matrix = None  # W matrix: [vocab_size, hidden_size]
        self.vocabulary = None
        self.token_regions = {}
        
    def extract_output_matrix(self) -> torch.Tensor:
        """Extract and cache the output projection matrix."""
        if self.output_matrix is None:
            self.output_matrix = self.model.get_output_projection_matrix()  # [vocab_size, hidden_size]
            self.vocabulary = self.model.get_vocabulary()
        return self.output_matrix
    
    def point_to_token_prediction(self, points: torch.Tensor) -> Dict[str, Any]:
        """
        Given points in embedding space, predict which tokens would be most likely.
        
        Args:
            points: Embedding vectors [batch_size, hidden_size] or [hidden_size] for a single point
            
        Returns:
            Dictionary with prediction info
        """
        W = self.extract_output_matrix()
        
        if points.dim() == 1:
            points = points.unsqueeze(0)  # [1, hidden_size]
            
        logits = torch.matmul(points, W.t())  # [batch_size, vocab_size]
        
        top_logits, top_token_ids = torch.max(logits, dim=1)  # [batch_size]
        
        top_tokens = [self.vocabulary.get(tid.item(), f"<unk_{tid.item()}>") for tid in top_token_ids]
        
        result = {
            'top_token_ids': top_token_ids.tolist(),
            'top_tokens': top_tokens,
            'top_logits': top_logits.tolist(),
            'all_logits': logits,
        }
        
        return result
    
    def check_point_in_region(self, point: torch.Tensor, token_id: int) -> bool:
        """
        Check if a point satisfies all constraints for a token's region.
        
        Args:
            point: Embedding vector [hidden_size]
            token_id: Token ID to check
            
        Returns:
            True if point is in the token's region
        """
        if token_id not in self.token_regions:
            return False
        
        region = self.token_regions[token_id]
        
        # Check all constraints: normal^T * point >= bias
        for normal, bias in region.constraints:
            if torch.dot(normal, point) < bias:
                return False
        
        return True
    
    def estimate_voronoi_volumes(
        self,
        radius: float,
        num_samples: int = 50000,
        subset_token_ids: Optional[List[int]] = None,
        chunk_size_tokens: int = 2048,
        point_batch_size: int = 8192,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Estimate Voronoi cell volumes (within a radius-R ball) for decoder token embeddings.

        Args:
            radius: Sphere radius that truncates the Voronoi cells.
            num_samples: Number of Monte Carlo samples used for the estimate.
            subset_token_ids: Optional list of token ids to restrict the computation.
            chunk_size_tokens: Token chunk size for distance computation.
            point_batch_size: Sample batch size for distance computation.
            seed: Optional RNG seed for reproducibility.
            device: Torch device string; defaults to CUDA if available.
        """
        W = self.extract_output_matrix()  # [vocab_size, hidden_size]
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = estimate_voronoi_cell_volumes(
            decoder_matrix=W,
            radius=radius,
            subset_indices=subset_token_ids,
            num_samples=num_samples,
            tokens_on_columns=False,
            device=dev,
            chunk_size_tokens=chunk_size_tokens,
            point_batch_size=point_batch_size,
            seed=seed,
        )

        # Attach token strings for convenience
        if self.vocabulary is None:
            self.extract_output_matrix()
        volumes_with_tokens = {}
        for tid in result["selected_token_ids"]:
            token_str = self.vocabulary.get(tid, f"<unk_{tid}>")
            volumes_with_tokens[tid] = {
                "token": token_str,
                "volume": result["volume_per_token"][tid],
                "fraction": result["proportion_per_token"][tid],
                "samples": result["counts"][tid],
            }
        result["volumes_with_tokens"] = volumes_with_tokens
        return result
    
    
