from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from simpler_fine_bert.common.managers import get_metrics_manager

# Get manager instance
metrics_manager = get_metrics_manager()

logger = logging.getLogger(__name__)

class SimCSEEmbeddingLoss(nn.Module):
    """Combined loss for embedding learning and SimCSE objectives."""
    
    def __init__(self, temperature: float = 0.05, embedding_weight: float = 1.0, simcse_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.embedding_weight = embedding_weight
        self.simcse_weight = simcse_weight
        
    def forward(self, outputs1: Dict[str, torch.Tensor], outputs2: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate combined embedding and SimCSE loss.
        
        Args:
            outputs1: First forward pass outputs
            outputs2: Second forward pass outputs (different dropout mask)
            labels: Token labels
            
        Returns:
            Dictionary containing loss and metrics
        """
        try:
            # Get embeddings
            embeddings1 = outputs1['pooler_output']  # [batch_size, hidden_size]
            embeddings2 = outputs2['pooler_output']  # [batch_size, hidden_size]
            
            # Normalize embeddings
            embeddings1 = nn.functional.normalize(embeddings1, p=2, dim=1)
            embeddings2 = nn.functional.normalize(embeddings2, p=2, dim=1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
            
            # Get positive pairs (diagonal)
            pos_sim = torch.diag(sim_matrix)
            
            # Compute SimCSE loss
            labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
            simcse_loss = nn.functional.cross_entropy(sim_matrix, labels)
            
            # Compute embedding loss
            embedding_loss1 = metrics_manager.compute_embedding_metrics(outputs1, {'labels': labels})['embedding_loss']
            embedding_loss2 = metrics_manager.compute_embedding_metrics(outputs2, {'labels': labels})['embedding_loss']
            embedding_loss = (embedding_loss1 + embedding_loss2) / 2
            
            # Combined loss
            total_loss = self.embedding_weight * embedding_loss + self.simcse_weight * simcse_loss
            
            # Get embedding metrics
            embedding_metrics1 = metrics_manager.compute_embedding_metrics(outputs1, {'labels': labels})
            embedding_metrics2 = metrics_manager.compute_embedding_metrics(outputs2, {'labels': labels})
            
            return {
                'loss': total_loss,
                'embedding_loss': embedding_loss,
                'simcse_loss': simcse_loss,
                'accuracy': (embedding_metrics1['accuracy'] + embedding_metrics2['accuracy']) / 2,
                'perplexity': (embedding_metrics1['perplexity'] + embedding_metrics2['perplexity']) / 2,
                'positive_similarity': pos_sim.mean(),
                'negative_similarity': (sim_matrix.sum() - pos_sim.sum()) / (sim_matrix.size(0) * (sim_matrix.size(0) - 1))
            }
            
        except Exception as e:
            logger.error(f"Error in SimCSEEmbeddingLoss forward pass: {str(e)}")
            raise

class FocalLoss(nn.Module):
    """Focal Loss for classification to focus on hard examples."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()
