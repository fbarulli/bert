from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import math

from simpler_fine_bert.common.managers.base_manager import BaseManager
from simpler_fine_bert.common.managers import get_cuda_manager

logger = logging.getLogger(__name__)

class MetricsManager(BaseManager):
    """Manager for computing and tracking metrics."""
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        # Call parent's initialization first
        super()._initialize_process_local(config)
        
        # Get cuda_manager at runtime
        cuda_manager = get_cuda_manager()
        
        # Initialize cuda_manager first since we depend on it
        cuda_manager.ensure_initialized()
        self._local.device = None
        self._local.loss_fct = None
    
    def get_device(self) -> torch.device:
        """Get current device."""
        self.ensure_initialized()
        if self._local.device is None:
            cuda_manager = get_cuda_manager()
            if cuda_manager.is_available():
                self._local.device = torch.device('cuda')
            else:
                self._local.device = torch.device('cpu')
        return self._local.device

    def get_loss_fct(self) -> nn.Module:
        """Get or create loss function."""
        self.ensure_initialized()
        if self._local.loss_fct is None:
            self._local.loss_fct = nn.CrossEntropyLoss(ignore_index=-100).to(self.get_device())
        return self._local.loss_fct
    
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
            
            # Calculate loss on masked tokens only
            loss = self.get_loss_fct()(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Get predictions and compute accuracy
            predictions = logits.argmax(dim=-1)  # [batch_size, seq_len]
            mask = labels != -100  # [batch_size, seq_len]
            
            # Compute accuracy only on masked tokens
            correct = (predictions[mask] == labels[mask]).float().sum()
            total = mask.sum()
            accuracy = (correct / total).item() if total > 0 else 0.0
            
            # Compute top-k accuracy
            _, top_k = logits.topk(5, dim=-1)  # [batch_size, seq_len, 5]
            correct_k = labels.unsqueeze(-1).expand_as(top_k) == top_k
            correct_k = correct_k & mask.unsqueeze(-1)
            top5_accuracy = (correct_k.any(dim=-1).float().sum() / total).item() if total > 0 else 0.0
            
            # Log detailed stats for debugging
            logger.debug(
                f"Metrics Stats:\n"
                f"- Total tokens: {labels.numel()}\n"
                f"- Masked tokens: {total.item()}\n"
                f"- Correct predictions: {correct.item()}\n"
                f"- Raw loss: {loss.item():.4f}\n"
                f"- Accuracy: {accuracy:.4%}\n"
                f"- Top-5 Accuracy: {top5_accuracy:.4%}"
            )
            
            # Calculate perplexity
            try:
                ppl = math.exp(loss.item())
                logger.debug(f"Calculated perplexity: exp({loss.item()}) = {ppl:.2f}")
            except OverflowError:
                logger.warning(f"Overflow computing perplexity for loss {loss.item()}")
                ppl = float('inf')
            
            metrics = {
                'loss': loss.item(),
                'embedding_loss': loss.item(),
                'ppl': ppl,  # Use consistent key name
                'accuracy': accuracy,
                'top5_accuracy': top5_accuracy
            }
            
            logger.debug(
                f"Embedding Metrics:\n"
                f"- Loss: {metrics['loss']:.4f}\n"
                f"- Perplexity: {metrics['ppl']:.2f}\n"
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
                'ppl': float('inf'),
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
            normalized_loss = self.get_loss_fct()(logits, labels)
            
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
        """Compute perplexity from loss without capping."""
        try:
            return math.exp(loss.item())
        except OverflowError:
            logger.warning(f"Overflow computing perplexity for loss {loss.item()}")
            return float('inf')

metrics_manager = MetricsManager()

__all__ = ['MetricsManager', 'metrics_manager']
