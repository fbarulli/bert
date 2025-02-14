from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # needed for F.normalize and others
import logging
import traceback  # for error handling
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy.typing as npt

from simpler_fine_bert.common.cuda_utils import tensor_manager
from simpler_fine_bert.common.cuda_manager import cuda_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean',
        contrast_mode: str = 'all',
        chunk_size: int = 256  # Process similarity matrix in chunks
    ) -> None:
        logger.info("Initializing InfoNCE Loss")
        try:
            super().__init__()
            self.temperature = temperature
            self.reduction = reduction
            self.contrast_mode = contrast_mode
            self.chunk_size = chunk_size
            logger.info(
                f"InfoNCE Loss initialized with temperature={temperature}, "
                f"reduction={reduction}, contrast_mode={contrast_mode}"
            )
        except Exception as e:
            logger.error(f"Error initializing InfoNCE Loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def compute_similarity_chunk(
        self,
        features: torch.Tensor,
        chunk_start: int,
        chunk_size: int
    ) -> torch.Tensor:
        try:
            chunk_end = min(chunk_start + chunk_size, features.size(0))
            chunk_features = features[chunk_start:chunk_end]
            
            # Compute similarity between chunk and all features
            sim_chunk = torch.matmul(chunk_features, features.T)
            return sim_chunk
            
        except Exception as e:
            logger.error(f"Error computing similarity chunk: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE loss.
        
        Args:
            features: Input features
            labels: Optional labels for supervised contrastive learning
            mask: Optional mask for valid pairs
            
        Returns:
            Dictionary of loss and metrics
        """
        logger.debug("Computing InfoNCE loss")
        try:
            # Ensure all inputs are on same device
            device = features.device
            if labels is not None:
                labels = labels.to(device)
            if mask is not None:
                mask = mask.to(device)
            
            # Normalize features and compute similarity metrics
            features = F.normalize(features, dim=1)
            batch_size = features.size(0)
            
            total_loss = 0.0
            total_pairs = 0
            total_pos_sim = 0.0
            total_neg_sim = 0.0
            
            # Process in chunks to save memory
            for i in range(0, batch_size, self.chunk_size):
                chunk_start = i
                chunk_end = min(i + self.chunk_size, batch_size)
                chunk_size = chunk_end - chunk_start
                
                # Get chunk features and labels
                chunk_features = features[chunk_start:chunk_end]
                chunk_labels = labels[chunk_start:chunk_end] if labels is not None else None
                
                try:
                    # Compute similarity with gradient clipping
                    chunk_features = torch.clamp(chunk_features, min=-1e3, max=1e3)
                    features_clipped = torch.clamp(features, min=-1e3, max=1e3)
                    sim_chunk = torch.matmul(chunk_features, features_clipped.T)
                    
                    # Apply temperature with stability check
                    temperature = max(self.temperature, 1e-4)  # Prevent division by zero
                    sim_chunk = sim_chunk / temperature
                    
                    # Create chunk masks on correct device
                    device = chunk_features.device
                    chunk_mask_self = torch.ones_like(sim_chunk, dtype=torch.bool, device=device)
                    chunk_mask_self[:, chunk_start:chunk_end].fill_diagonal_(False)
                    
                    if chunk_labels is not None:
                        chunk_labels = chunk_labels.contiguous().view(-1, 1).to(device)
                        chunk_mask_pos = chunk_labels == labels.view(1, -1).to(device)
                        chunk_mask_pos = chunk_mask_pos & chunk_mask_self
                        # Ensure at least one positive pair
                        if not chunk_mask_pos.any():
                            chunk_mask_pos = chunk_mask_self
                    else:
                        chunk_mask_pos = chunk_mask_self
                    
                    # For numerical stability
                    sim_max, _ = torch.max(sim_chunk, dim=1, keepdim=True)
                    sim_chunk = sim_chunk - sim_max.detach()
                    sim_chunk = torch.clamp(sim_chunk, min=-1e3, max=1e3)
                    
                    # Compute exp and log-sum-exp with stability
                    exp_sim = torch.exp(sim_chunk)
                    exp_sim = torch.clamp(exp_sim, min=1e-8)  # Prevent zero exp
                    exp_sim = exp_sim * chunk_mask_self
                    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
                    
                    # Compute log probabilities
                    log_prob = sim_chunk - log_sum_exp
                    log_prob = torch.clamp(log_prob, min=-1e3)  # Prevent -inf
                    
                    # Compute mean of positive pairs for chunk
                    pos_pairs = chunk_mask_pos.sum(1)
                    chunk_loss = -(chunk_mask_pos * log_prob).sum(1)
                    valid_pairs = pos_pairs > 0
                    if valid_pairs.any():
                        chunk_loss = chunk_loss[valid_pairs] / pos_pairs[valid_pairs]
                    else:
                        chunk_loss = torch.zeros(1, device=device)
                except Exception as e:
                    logger.error(f"Error in chunk computation: {str(e)}")
                    chunk_loss = torch.zeros(1, device=device)
                
                # Accumulate loss
                total_loss += chunk_loss.sum()
                total_pairs += (pos_pairs > 0).sum()
            
            # Compute mean loss
            mean_loss = total_loss / (total_pairs + 1e-8)
            
            # Handle reduction
            if self.reduction == 'mean':
                loss = mean_loss
            elif self.reduction == 'sum':
                loss = mean_loss * total_pairs
            else:
                loss = mean_loss
            
            # Compute average similarities
            avg_pos_sim = total_pos_sim / (total_pairs + 1e-8)
            avg_neg_sim = total_neg_sim / (total_pairs + 1e-8)
            
            return {
                'loss': loss,
                'positive_similarity': avg_pos_sim,
                'negative_similarity': avg_neg_sim,
                'num_pairs': total_pairs,
                'mean_loss': mean_loss
            }
            
        except Exception as e:
            logger.error(f"Error computing InfoNCE loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class SimCSEEmbeddingLoss(nn.Module):
    """SimCSE loss for learning sentence embeddings.
    
    SimCSE uses dropout noise as data augmentation to create positive pairs
    from the same input sentence. The model processes each input twice with
    different dropout masks to get two different embeddings of the same sentence.
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        reduction: str = 'mean',
        use_amp: bool = True
    ) -> None:
        """Initialize SimCSE loss.
        
        Args:
            temperature: Temperature parameter for similarity scaling
            reduction: Reduction method for loss ('mean' or 'sum')
            use_amp: Whether to use automatic mixed precision
        """
        logger.info("Initializing SimCSE Loss")
        try:
            super().__init__()
            self.temperature = temperature
            self.reduction = reduction
            self.use_amp = use_amp
            logger.info(
                f"SimCSE Loss initialized with temperature={temperature}, "
                f"reduction={reduction}, use_amp={use_amp}"
            )
        except Exception as e:
            logger.error(f"Error initializing SimCSE Loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute SimCSE loss using dropout for positive pairs.
        
        Args:
            embeddings: Input embeddings [batch_size * 2, hidden_size]
                       First half and second half are embeddings of same sentences
                       with different dropout masks
            attention_mask: Optional attention mask [batch_size * 2, seq_len]
            
        Returns:
            Dictionary containing loss and metrics
        """
        logger.debug("Computing SimCSE loss")
        try:
            # Split embeddings into original and augmented views
            batch_size = embeddings.size(0) // 2
            z1, z2 = embeddings[:batch_size], embeddings[batch_size:]
            
            # Normalize embeddings
            with cuda_manager.amp.autocast(enabled=self.use_amp):
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                
                # Compute similarity matrix
                sim = torch.matmul(z1, z2.T) / self.temperature
                
                # Labels are diagonal (positive pairs)
                labels = torch.arange(batch_size, device=sim.device)
                
                # Compute loss
                loss = F.cross_entropy(sim, labels)
                
                # Compute accuracy
                pred = sim.argmax(dim=1)
                acc = (pred == labels).float().mean()
                
                # Compute average positive and negative similarities
                pos_sim = torch.diagonal(sim).mean()
                neg_sim = (sim.sum() - torch.diagonal(sim).sum()) / (batch_size * (batch_size - 1))
                
                metrics = {
                    'loss': loss,
                    'accuracy': acc,
                    'positive_similarity': pos_sim,
                    'negative_similarity': neg_sim
                }
                
                if self.reduction == 'sum':
                    metrics['loss'] = metrics['loss'] * batch_size
                    
                return metrics
                
        except Exception as e:
            logger.error(f"Error computing SimCSE loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class ContrastiveLearningWrapper:
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 0.07,
        queue_size: int = 65536,
        chunk_size: int = 256  # Process features in chunks
    ) -> None:
        logger.info("Initializing ContrastiveLearningWrapper")
        try:
            self.model = model
            self.criterion = InfoNCELoss(
                temperature=temperature,
                chunk_size=chunk_size
            )
            self.queue_size = queue_size
            self.chunk_size = chunk_size
            self.register_queue()
            logger.info("ContrastiveLearningWrapper initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ContrastiveLearningWrapper: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def register_queue(self) -> None:
        try:
            self.queue: Optional[torch.Tensor] = None
            self.queue_ptr = tensor_manager.create_tensor(torch.zeros(1), dtype=torch.long)
            logger.info(f"Queue initialized with size {self.queue_size}")
        except Exception as e:
            logger.error(f"Error initializing queue: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        try:
            batch_size = keys.shape[0]
            
            # Initialize queue if not exists
            if self.queue is None:
                self.queue = tensor_manager.create_tensor(
                    torch.zeros((self.queue_size, keys.shape[1])),
                    dtype=keys.dtype,
                    device=keys.device
                )
                self.queue_ptr[0] = 0
            
            # Get current pointer
            ptr = int(self.queue_ptr[0])
            
            # Compute how many samples we can add
            space_left = self.queue_size - ptr
            samples_to_add = min(batch_size, space_left)
            
            # Add samples to queue
            if samples_to_add > 0:
                self.queue[ptr:ptr + samples_to_add] = keys[:samples_to_add]
                ptr = (ptr + samples_to_add) % self.queue_size
                self.queue_ptr[0] = ptr
            
            # If queue is not full, don't use it yet
            if ptr < self.queue_size and self.queue_ptr[0] == 0:
                self.queue = None
            
        except Exception as e:
            logger.error(f"Error in dequeue and enqueue: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        logger.debug("Computing contrastive loss")
        try:
            # Process features in chunks if needed
            if features.size(0) > self.chunk_size:
                total_loss = 0.0
                num_chunks = (features.size(0) + self.chunk_size - 1) // self.chunk_size
                
                for i in range(num_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, features.size(0))
                    
                    chunk_features = features[start_idx:end_idx]
                    chunk_labels = labels[start_idx:end_idx] if labels is not None else None
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        # Get negative samples from queue
                        if self.queue is not None:
                            all_features = torch.cat([chunk_features, self.queue], dim=0)
                            if chunk_labels is not None:
                                all_labels = torch.cat([
                                    chunk_labels,
                                    tensor_manager.create_tensor(torch.zeros(self.queue_size), device=chunk_labels.device)
                                ])
                            else:
                                all_labels = None
                        else:
                            all_features = chunk_features
                            all_labels = chunk_labels
                        
                        # Compute loss and metrics for chunk
                        metrics = self.criterion(all_features, all_labels)
                        if torch.isfinite(metrics['loss']):
                            total_loss += metrics['loss'] * (end_idx - start_idx)
                        else:
                            logger.warning("Skipping invalid chunk loss")
                
                # Update queue with all features
                self._dequeue_and_enqueue(features)
                
                avg_loss = total_loss / features.size(0)
                return {
                    'loss': avg_loss,
                    'mean_loss': avg_loss,
                    'num_chunks': num_chunks
                }
            else:
                # Process entire batch at once
                if self.queue is not None:
                    all_features = torch.cat([features, self.queue], dim=0)
                    if labels is not None:
                        all_labels = torch.cat([
                            labels,
                            tensor_manager.create_tensor(torch.zeros(self.queue_size), device=labels.device)
                        ])
                    else:
                        all_labels = None
                else:
                    all_features = features
                    all_labels = labels
                
                # Compute loss and metrics
                metrics = self.criterion(all_features, all_labels)
                
                # Update queue
                self._dequeue_and_enqueue(features)
                
                return metrics
            
        except Exception as e:
            logger.error(f"Error computing contrastive loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise
