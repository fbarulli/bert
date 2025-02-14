from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import math

logger = logging.getLogger(__name__)

class MetricsManager:
    """Manager for computing and tracking metrics."""
    
    def __init__(self):
        """Initialize metrics manager."""
        self.initialized = False
        self.device = None
        self.loss_fct = None
        
    def initialize(self, device: torch.device):
        """Initialize metrics manager with device."""
        if not self.initialized:
            self.device = device
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100).to(device)
            self.initialized = True
            logger.debug(f"Initialized metrics manager on {device}")
    
    def ensure_initialized(self):
        """Ensure metrics manager is initialized."""
        if not self.initialized:
            raise RuntimeError("MetricsManager not initialized")
    
    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int = 1
    ) -> Dict[str, float]:
        """Compute top-k accuracy for masked tokens only.
        
        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: True labels [batch_size, seq_len]
            k: Top-k accuracy to compute
            
        Returns:
            Dictionary of accuracy metrics
        """
        self.ensure_initialized()
        
        # Only consider positions where labels != -100
        active_preds = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
        active_labels = labels.view(-1)  # [batch_size * seq_len]
        active_mask = active_labels != -100
        
        if not active_mask.any():
            logger.warning("No valid positions for accuracy calculation")
            return {'top1': 0.0, f'top{k}': 0.0}
        
        active_preds = active_preds[active_mask]  # [num_valid, vocab_size]
        active_labels = active_labels[active_mask]  # [num_valid]
        
        # Get top-k predictions
        _, pred_indices = active_preds.topk(k, dim=1)  # [num_valid, k]
        
        # Check if true label is in top-k predictions
        correct_k = pred_indices.eq(active_labels.unsqueeze(1).expand_as(pred_indices))
        
        # Calculate accuracies
        total = active_mask.sum().item()
        top1 = correct_k[:, 0].sum().item() / total
        topk = correct_k.any(dim=1).sum().item() / total
        
        logger.debug(
            f"Accuracy Stats:\n"
            f"- Total predictions: {total}\n"
            f"- Top-1 correct: {int(top1 * total)}\n"
            f"- Top-{k} correct: {int(topk * total)}"
        )
        
        return {'top1': top1, f'top{k}': topk}
    
    def compute_embedding_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute embedding metrics with proper loss normalization.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Dictionary of metrics
        """
        self.ensure_initialized()
        
        try:
            # Get logits and labels
            logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
            labels = batch['labels']  # [batch_size, seq_len]
            
            # Calculate loss
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Normalize loss by number of masked tokens
            num_masked = (labels != -100).sum().item()
            if num_masked > 0:
                normalized_loss = loss * labels.size(0) * labels.size(1) / num_masked
            else:
                normalized_loss = loss
            
            # Compute accuracy metrics
            accuracy_metrics = self.compute_accuracy(logits, labels, k=5)
            
            metrics = {
                'loss': normalized_loss.item(),  # Use normalized loss for both
                'embedding_loss': normalized_loss.item(),  # Keep both for compatibility
                'perplexity': self.compute_perplexity(normalized_loss),
                'accuracy': accuracy_metrics['top1'],
                'top5_accuracy': accuracy_metrics['top5']
            }
            
            logger.debug(
                f"Embedding Metrics:\n"
                f"- Loss: {metrics['loss']:.4f}\n"
                f"- Perplexity: {metrics['perplexity']:.2f}\n"
                f"- Accuracy: {metrics['accuracy']:.4f}\n"
                f"- Top-5 Accuracy: {metrics['top5_accuracy']:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing embedding metrics: {str(e)}")
            # Return default metrics in case of error
            return {
                'loss': float('inf'),
                'embedding_loss': float('inf'),
                'perplexity': float('inf'),
                'accuracy': 0.0,
                'top5_accuracy': 0.0
            }
    
    def compute_classification_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Dictionary of metrics
        """
        self.ensure_initialized()
        
        try:
            # Get logits and labels
            logits = outputs['logits']  # [batch_size, num_classes]
            labels = batch['labels']  # [batch_size]
            
            # Calculate loss
            normalized_loss = self.loss_fct(logits, labels)
            
            # Get predictions
            _, preds = logits.max(dim=1)
            
            # Calculate accuracy
            correct = preds.eq(labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0
            
            metrics = {
                'loss': normalized_loss.item(),
                'accuracy': accuracy
            }
            
            logger.debug(
                f"Classification Metrics:\n"
                f"- Loss: {metrics['loss']:.4f}\n"
                f"- Accuracy: {metrics['accuracy']:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing classification metrics: {str(e)}")
            return {
                'loss': float('inf'),
                'accuracy': 0.0
            }
    
    def compute_perplexity(self, loss: torch.Tensor) -> float:
        """Compute perplexity from loss."""
        try:
            return math.exp(min(20, loss.item()))
        except OverflowError:
            return float('inf')

metrics_manager = MetricsManager()

__all__ = ['MetricsManager', 'metrics_manager']
