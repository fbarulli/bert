from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import traceback
from scipy.stats import spearmanr
import umap
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy.typing as npt

from simpler_fine_bert.cuda_utils import tensor_manager, metrics_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingAnalyzer:
    def __init__(self, device: Optional[torch.device] = None, wandb_manager: Optional[Any] = None) -> None:
        logger.info("Initializing EmbeddingAnalyzer")
        try:
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.wandb_manager = wandb_manager
            self.metrics_history: Dict[str, List[float]] = defaultdict(list)
            logger.info(f"EmbeddingAnalyzer initialized successfully on device {self.device}")
        except Exception as e:
            logger.error(f"Error initializing EmbeddingAnalyzer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_embedding_space(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        logger.info("Starting embedding space analysis")
        try:
            metrics: Dict[str, float] = {}
            
            if self.wandb_manager:
                self.wandb_manager.log_analysis_progress("Starting analysis", 0)
            
            # Compute basic embedding metrics
            basic_metrics = metrics_manager.compute_embedding_metrics(embeddings)
            metrics.update({
                'isotropy': basic_metrics['isotropy'],
                'avg_cosine_sim': basic_metrics['avg_cosine_sim'],
                'cosine_sim_std': basic_metrics['cosine_sim_std'],
                'norm_mean': basic_metrics['norm_mean'],
                'norm_std': basic_metrics['norm_std'],
                'effective_dim': basic_metrics['effective_dim']
            })
            
            if self.wandb_manager:
                self.wandb_manager.log_analysis_progress("Basic metrics computed", 50)
            
            if labels is not None:
                # Compute label-based metrics
                label_metrics = metrics_manager.compute_label_metrics(embeddings, labels)
                metrics.update({
                    'intra_class_sim': label_metrics['intra_class_sim'],
                    'inter_class_sim': label_metrics['inter_class_sim'],
                    'cluster_label_alignment': label_metrics['cluster_alignment'],
                    'class_separability': label_metrics['separability']
                })
            
            if self.wandb_manager:
                self.wandb_manager.log_analysis_progress("Analysis completed", 100)
            
            # Store metrics history
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
            
            logger.info("Embedding space analysis completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in embedding space analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def visualize_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        method: str = 'tsne',
        save_path: Optional[str] = None,
        perplexity: int = 30,
        n_neighbors: int = 15
    ) -> None:
        logger.info(f"Starting embedding visualization using {method}")
        try:
            # Move embeddings to CPU for numpy conversion
            emb_np: npt.NDArray[np.float32] = embeddings.detach().cpu().numpy()
            if labels is not None:
                labels = labels.detach().cpu()
            
            if self.wandb_manager:
                self.wandb_manager.log_analysis_progress(f"Starting {method} reduction", 0)
                if method.lower() == 'tsne':
                    reducer = TSNE(
                        n_components=2,
                        perplexity=perplexity,
                        n_iter=1000,
                        verbose=0
                    )
                else:  # UMAP
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        verbose=0
                    )
                
                reduced_embeddings = reducer.fit_transform(emb_np)
                if self.wandb_manager:
                    self.wandb_manager.log_analysis_progress(f"{method} reduction completed", 100)
            
            plt.figure(figsize=(10, 8))
            if labels is not None:
                labels_np = labels.numpy()
                scatter = plt.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    c=labels_np,
                    cmap='tab10',
                    alpha=0.6
                )
                plt.colorbar(scatter)
            else:
                plt.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    alpha=0.6
                )
            
            plt.title(f'Embedding Visualization ({method.upper()})')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Visualization saved to {save_path}")
            plt.close()
            
            logger.info("Embedding visualization completed successfully")
            
        except Exception as e:
            logger.error(f"Error in embedding visualization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def plot_metrics_history(
        self,
        save_path: Optional[str] = None
    ) -> None:
        logger.info("Starting metrics history visualization")
        try:
            n_metrics = len(self.metrics_history)
            fig, axes = plt.subplots(
                (n_metrics + 1) // 2, 2,
                figsize=(15, 5 * ((n_metrics + 1) // 2))
            )
            axes = axes.flatten()
            
            for (metric_name, metric_values), ax in zip(self.metrics_history.items(), axes):
                ax.plot(metric_values, marker='o')
                ax.set_title(f'{metric_name} Over Time')
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name)
                ax.grid(True)
            
            # Remove empty subplots if odd number of metrics
            if n_metrics % 2 != 0:
                fig.delaxes(axes[-1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Metrics history plot saved to {save_path}")
            plt.close()
            
            logger.info("Metrics history visualization completed successfully")
            
        except Exception as e:
            logger.error(f"Error in metrics history visualization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def compute_similarity_matrix(
        self,
        embeddings: torch.Tensor,
        batch_size: int = 128
    ) -> torch.Tensor:
        logger.info("Computing similarity matrix")
        try:
            # Compute similarity matrix using metrics manager
            similarity_metrics = metrics_manager.compute_similarity_matrix(
                embeddings,
                batch_size=batch_size,
                progress_callback=lambda p: self.wandb_manager.log_analysis_progress("Computing similarity matrix", p) if self.wandb_manager else None
            )
            
            logger.info("Similarity matrix computation completed successfully")
            return similarity_metrics['similarity_matrix']
            
        except Exception as e:
            logger.error(f"Error in similarity matrix computation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
