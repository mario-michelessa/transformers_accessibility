from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .token_partition_analyzer import TokenPartitionAnalyzer


class ProjectionHandler:
    """Handles dimensionality reduction for visualization."""
    
    def __init__(self):
        self.pca = None
        
    def fit_projection(self, embeddings: torch.Tensor, **kwargs) -> None:
        """Fit dimensionality reduction for visualization."""
        embeddings_np = embeddings.detach().numpy()

        self.pca = PCA(n_components=2, **kwargs)
        self.pca.fit(embeddings_np)

    def project_to_2d(self, embeddings: torch.Tensor) -> np.ndarray:
        """Project embeddings to 2D using fitted projection."""
        embeddings_np = embeddings.detach().numpy()

        if self.pca is not None:
            return self.pca.transform(embeddings_np)
    
    def inverse_transform_2d(self, points_2d: np.ndarray) -> Optional[torch.Tensor]:
        """Map 2D points back to high-dimensional space (PCA only)."""
        if self.pca is not None:
            points_hd = self.pca.inverse_transform(points_2d)
            return torch.from_numpy(points_hd).float()
        return None


class GridOperations:
    """Handles 2D grid creation and coordinate transformations."""
    
    @staticmethod
    def create_2d_grid(bounds: Tuple[float, float, float, float], resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Create a 2D grid for visualization."""
        xmin, xmax, ymin, ymax = bounds
        x = np.linspace(xmin, xmax, resolution)
        y = np.linspace(ymin, ymax, resolution)
        return np.meshgrid(x, y)
    
    @staticmethod
    def calculate_bounds(points_2d: np.ndarray, margin: float = 3.0, multiplier: float = 1.0) -> Tuple[float, float, float, float]:
        """Calculate bounds for visualization plane."""
        xmin = (points_2d.min() - margin) * multiplier
        xmax = (points_2d.max() + margin) * multiplier
        ymin = (points_2d.min() - margin) * multiplier
        ymax = (points_2d.max() + margin) * multiplier
        # xmin = (points_2d[:, 0].min() - margin) * multiplier
        # xmax = (points_2d[:, 0].max() + margin) * multiplier
        # ymin = (points_2d[:, 1].min() - margin) * multiplier
        # ymax = (points_2d[:, 1].max() + margin) * multiplier
        return xmin, xmax, ymin, ymax


class TokenPredictionEngine:
    """Handles token predictions on 2D planes."""
    
    def __init__(self, analyzer: TokenPartitionAnalyzer, projection_handler: ProjectionHandler):
        self.analyzer = analyzer
        self.projection_handler = projection_handler
    
    def predict_tokens_on_plane(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Predict tokens for each pixel on the 2D plane."""
        resolution = X.shape[0]
        predictions = np.zeros_like(X, dtype=int)
        
        # Convert 2D grid back to high-dimensional embedding space
        points_2d = np.column_stack([X.ravel(), Y.ravel()])
        points_hd = self.projection_handler.inverse_transform_2d(points_2d)
        
        if points_hd is None:
            raise NotImplementedError("Inverse projection not available for this method")
        
        # Batch predict tokens for efficiency
        total_points = resolution * resolution
        
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_points = points_hd[batch_start:batch_end]

            batch_predictions = self.analyzer.point_to_token_prediction(batch_points)

            # Map back to 2D coordinates
            for i, token_id in enumerate(batch_predictions['top_token_ids']):
                flat_idx = batch_start + i
                row = flat_idx // resolution
                col = flat_idx % resolution
                predictions[row, col] = token_id
        
        return predictions


class VisualizationRenderer:
    """Handles matplotlib rendering and styling."""
    
    def __init__(self, analyzer: TokenPartitionAnalyzer):
        self.analyzer = analyzer
    
    def create_token_region_plot(self, X: np.ndarray, Y: np.ndarray, token_predictions: np.ndarray, 
                                text_embeddings_2d: np.ndarray, texts: List[str],
                                bounds: Tuple[float, float, float, float], max_colors: int = 50,
                                figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """Create the main token region visualization plot."""
        xmin, xmax, ymin, ymax = bounds
        resolution = X.shape[0]
        
        # Process token colors
        unique_tokens, token_counts = np.unique(token_predictions, return_counts=True)
        sorted_indices = np.argsort(token_counts)[::-1]
        top_tokens = unique_tokens[sorted_indices[:max_colors]]
        
        color_array = np.full_like(token_predictions, -1, dtype=int)
        for i, token_id in enumerate(top_tokens):
            color_array[token_predictions == token_id] = i
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        user_colors = [
            (31, 119, 180, 255),   # Blue
            (255, 127, 14, 255),   # Orange
            (44, 160, 44, 255),    # Green
            (214, 39, 40, 255),    # Red
            (148, 103, 189, 255),  # Purple
            (140, 86, 75, 255),    # Brown
            (227, 119, 194, 255),  # Pink
            (127, 127, 127, 255),  # Gray
            (188, 189, 34, 255),   # Olive
            (23, 190, 207, 255),   # Cyan
            (174, 199, 232, 255),  # Light Blue
            (255, 187, 120, 255),  # Light Orange
            (152, 223, 138, 255),  # Light Green
            (255, 152, 150, 255),  # Light Red
            (197, 176, 213, 255),  # Light Purple
            (196, 156, 148, 255),  # Light Brown
            (247, 182, 210, 255),  # Light Pink
            (199, 199, 199, 255),  # Light Gray
            (219, 219, 141, 255),  # Light Olive
            (158, 218, 229, 255)   # Light Cyan
        ]
        if user_colors:
            def _normalize_color(col):
                arr = np.asarray(col, dtype=float)
                if arr.max() > 1.0: arr = arr / 255.0
                return arr.tolist()

            normalized = np.vstack([_normalize_color(c) for c in user_colors])
            colors = normalized[: len(top_tokens)]
            colors = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
            cmap = ListedColormap(colors)
            print(cmap.colors)
        else:
            # Fallback to the previous colormap behavior
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_tokens)))
            colors = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
            cmap = ListedColormap(colors)
        
        # Plot colored regions
        ax.imshow(color_array, extent=[xmin, xmax, ymin, ymax], 
                 origin='lower', cmap=cmap, alpha=0.8)
        
        # Add text embeddings
        ax.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
                  c='red', s=200, marker='o', linewidths=3, 
                  edgecolors='white', zorder=10, label='Text Embeddings')
        
        # Add annotations
        for i, (x, y) in enumerate(text_embeddings_2d):
            ax.annotate(f"Text {i+1}:\n'{texts[i][:20]}...'", 
                       (x, y), xytext=(10, 10), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                       zorder=11)
        
        # Add legend
        self._add_token_legend(ax, top_tokens, token_counts, sorted_indices, colors, unique_tokens, max_colors)
        
        ax.set_title(f'Token Prediction Regions on PCA Plane\n{resolution}x{resolution} resolution, {len(unique_tokens)} unique tokens')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _add_token_legend(self, ax, top_tokens, token_counts, sorted_indices, colors, unique_tokens, max_colors):
        """Add legend for token colors."""
        vocab = self.analyzer.vocabulary
        legend_elements = []
        
        for i, token_id in enumerate(top_tokens[:10]):
            token_str = vocab.get(token_id, f'<unk_{token_id}>')
            count = token_counts[sorted_indices[i]]
            legend_elements.append(
                patches.Rectangle((0,0),1,1, facecolor=colors[i], 
                                label=f"'{token_str}' ({count:,} pixels)")
            )
        
        if len(unique_tokens) > max_colors:
            remaining = len(unique_tokens) - max_colors
            legend_elements.append(
                patches.Rectangle((0,0),1,1, facecolor='gray', 
                                label=f'Other tokens ({remaining})')
            )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
                 fontsize=9, title='Next Token Predictions')
    
class EmbeddingSpaceVisualizer:
    """Main visualizer orchestrating all visualization components."""
    
    def __init__(self, analyzer: TokenPartitionAnalyzer):
        self.analyzer = analyzer
        self.projection_handler = ProjectionHandler()
        self.token_prediction_engine = TokenPredictionEngine(analyzer, self.projection_handler)
        self.renderer = VisualizationRenderer(analyzer)
    
    def fit_projection(self, embeddings: torch.Tensor, method: str = "pca", **kwargs) -> None:
        """Fit dimensionality reduction for visualization."""
        self.projection_handler.fit_projection(embeddings, **kwargs)
    
    def project_to_2d(self, embeddings: torch.Tensor) -> np.ndarray:
        """Project embeddings to 2D using fitted projection."""
        return self.projection_handler.project_to_2d(embeddings)

    def _extract_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Extract embeddings from texts."""
        text_embeddings = []
        for i, text in enumerate(texts):
            embedding = self.analyzer.model.extract_last_token_embedding(text, before_decoder=True)
            text_embeddings.append(embedding.squeeze())
            print(f"  Text {i+1}: '{text[:50]}...' -> embedding shape: {embedding.shape}")
        return torch.stack(text_embeddings)
    
    def plot_token_regions_from_texts(self, texts: List[str], 
                                      text_embeddings: Optional[torch.Tensor] = None,
                                      resolution: int = 100, 
                                    figsize: Tuple[int, int] = (15, 8),
                                    max_colors: int = 50, 
                                    plane_radius_multiplier: float = 1.0) -> plt.Figure:
        """Create visualization based on embeddings from specific texts using PCA plane."""
        if len(texts) != 3:
            raise ValueError("Exactly 3 texts required to define PCA plane")

        if text_embeddings is None:
            text_embeddings = self._extract_text_embeddings(texts)
        print(f"🎨 Creating PCA plane visualization from {len(texts)} texts...")
        
        # Extract and fit embeddings
        self.fit_projection(text_embeddings, method="pca")
        text_embeddings_2d = self.project_to_2d(text_embeddings)
        
        # Calculate bounds and create grid
        bounds = GridOperations.calculate_bounds(text_embeddings_2d, margin=3.0, multiplier=plane_radius_multiplier)
        X, Y = GridOperations.create_2d_grid(bounds, resolution)
        
        print(f"  📏 Plane bounds: x=[{bounds[0]:.2f}, {bounds[1]:.2f}], y=[{bounds[2]:.2f}, {bounds[3]:.2f}]")
        print(f"  🔢 Created {resolution}x{resolution} grid ({resolution*resolution:,} pixels)")
        
        # Predict tokens on the plane
        print("  🎯 Computing token predictions for each pixel...")
        token_predictions = self.token_prediction_engine.predict_tokens_on_plane(X, Y)
        
        # Create and return the visualization
        return self.renderer.create_token_region_plot(
            X, Y, token_predictions, text_embeddings_2d, texts, bounds, max_colors, figsize
        )
    