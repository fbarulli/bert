from __future__ import annotations

import torch
import torch.nn as nn
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    NeuronConductance,
    LayerActivation,
    LayerConductance,
    InternalInfluence,
    NoiseTunnel
)
from captum.attr._utils.visualization import visualize_text
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from simpler_fine_bert.common import get_tensor_manager
from simpler_fine_bert.common.metrics_manager import metrics_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingInterpreter:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        wandb_manager: Optional[Any] = None,
        worker_id: Optional[int] = None
    ) -> None:
        logger.info("Initializing EmbeddingInterpreter")
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.wandb_manager = wandb_manager
            self.worker_id = worker_id
            
            # Initialize Captum attribution methods
            self.integrated_gradients = LayerIntegratedGradients(
                self.forward_func,
                self.model.bert.embeddings
            )
            
            self.neuron_conductor = NeuronConductance(self.forward_func)
            self.layer_conductor = LayerConductance(
                self.forward_func,
                self.model.bert.encoder.layer[-1]
            )
            
            logger.info("EmbeddingInterpreter initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing EmbeddingInterpreter: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward_func(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        try:
            outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs['logits']
        except Exception as e:
            logger.error(f"Error in forward function: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_token_attributions(
        self,
        text: str,
        target_label: int,
        n_steps: int = 50,
        internal_batch_size: int = 5
    ) -> Dict[str, Any]:
        logger.info("Analyzing token attributions")
        try:
            # Tokenize input and move to device
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(tokens, device=self.device)
            
            # Get attributions
            attributions = self.integrated_gradients.attribute(
                inputs=tokens,
                target=target_label,
                additional_forward_args=(attention_mask,),
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )
            
            # Process attributions using metrics manager
            attribution_metrics = metrics_manager.compute_attribution_metrics(attributions)
            
            # Get word tokens for visualization
            word_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
            
            return {
                'tokens': word_tokens,
                'attributions': attribution_metrics['normalized_attributions'],
                'raw_attributions': attribution_metrics['raw_attributions'],
                'attribution_stats': {
                    'mean': attribution_metrics['mean'],
                    'std': attribution_metrics['std'],
                    'max': attribution_metrics['max'],
                    'min': attribution_metrics['min']
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing token attributions: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_neuron_behavior(
        self,
        text: str,
        layer_idx: int,
        neuron_idx: int
    ) -> Dict[str, Any]:
        logger.info(f"Analyzing neuron behavior for layer {layer_idx}, neuron {neuron_idx}")
        try:
            # Tokenize input and move to device
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(tokens, device=self.device)
            
            # Get neuron attributions
            neuron_attrs = self.neuron_conductor.attribute(
                inputs=tokens,
                neuron_selector=(layer_idx, neuron_idx),
                additional_forward_args=(attention_mask,)
            )
            
            # Process attributions using metrics manager
            neuron_metrics = metrics_manager.compute_neuron_metrics(neuron_attrs)
            word_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
            
            return {
                'tokens': word_tokens,
                'neuron_attributions': neuron_metrics['attributions'],
                'raw_attributions': neuron_metrics['raw_attributions'],
                'neuron_stats': {
                    'mean_activation': neuron_metrics['mean_activation'],
                    'peak_activation': neuron_metrics['peak_activation'],
                    'activation_frequency': neuron_metrics['activation_frequency'],
                    'selectivity': neuron_metrics['selectivity']
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing neuron behavior: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_layer_influence(
        self,
        text: str,
        target_label: int
    ) -> Dict[str, Any]:
        logger.info("Analyzing layer influence")
        try:
            # Tokenize input and move to device
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(tokens, device=self.device)
            
            # Get layer attributions
            layer_attrs = self.layer_conductor.attribute(
                inputs=tokens,
                target=target_label,
                additional_forward_args=(attention_mask,)
            )
            
            # Process attributions using metrics manager
            layer_metrics = metrics_manager.compute_layer_metrics(layer_attrs)
            word_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
            
            return {
                'tokens': word_tokens,
                'layer_attributions': layer_metrics['attributions'],
                'raw_attributions': layer_metrics['raw_attributions'],
                'layer_stats': {
                    'mean_influence': layer_metrics['mean_influence'],
                    'influence_std': layer_metrics['influence_std'],
                    'top_neurons': layer_metrics['top_neurons'],
                    'attention_patterns': layer_metrics['attention_patterns']
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing layer influence: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def visualize_attributions(
        self,
        attributions_dict: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        logger.info("Visualizing attributions")
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot token attributions
            sns.barplot(
                x=list(range(len(attributions_dict['tokens']))),
                y=attributions_dict['attributions']
            )
            
            # Customize plot
            plt.xticks(
                range(len(attributions_dict['tokens'])),
                attributions_dict['tokens'],
                rotation=45,
                ha='right'
            )
            plt.xlabel('Tokens')
            plt.ylabel('Attribution Score')
            plt.title('Token-level Attribution Analysis')
            
            # Save or show
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing attributions: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_embedding_space(
        self,
        texts: List[str],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        logger.info("Analyzing embedding space")
        try:
            results = {}
            
            # Process texts in batches
            batch_size = 32
            all_embeddings = []
            all_attributions = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            if self.wandb_manager:
                self.wandb_manager.log_analysis_progress("Starting embedding analysis", 0)
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Get embeddings and move to device
                tokens = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    outputs = self.model(**tokens, return_dict=True)
                    embeddings = outputs['pooled_output']
                    all_embeddings.append(embeddings.cpu())
                
                # Get attributions for each text
                if labels is not None:
                    batch_labels = labels[i:i + batch_size]
                    attributions = self.integrated_gradients.attribute(
                        inputs=tokens['input_ids'],
                        target=batch_labels,
                        additional_forward_args=(tokens['attention_mask'],)
                    )
                    all_attributions.append(attributions.cpu())
                
                if self.wandb_manager:
                    progress = ((i + batch_size) / len(texts)) * 100
                    self.wandb_manager.log_analysis_progress("Processing embeddings", progress)
            
            # Combine results
            embeddings = torch.cat(all_embeddings, dim=0)
            results['embeddings'] = embeddings
            
            if all_attributions:
                attributions = torch.cat(all_attributions, dim=0)
                results['attributions'] = attributions
            
            # Compute embedding space metrics using metrics manager
            embedding_metrics = metrics_manager.compute_embedding_metrics(embeddings)
            results.update({
                'embedding_norm': embedding_metrics['norm'],
                'embedding_std': embedding_metrics['std'],
                'isotropy': embedding_metrics['isotropy'],
                'cosine_similarities': embedding_metrics['cosine_similarities'],
                'clustering_metrics': embedding_metrics['clustering_metrics']
            })
            
            if len(all_attributions) > 0:
                attribution_metrics = metrics_manager.compute_attribution_metrics(
                    torch.cat(all_attributions, dim=0)
                )
                results.update({
                    'attribution_magnitude': attribution_metrics['magnitude'],
                    'attribution_sparsity': attribution_metrics['sparsity'],
                    'attribution_distribution': attribution_metrics['distribution']
                })
            
            if self.wandb_manager:
                self.wandb_manager.log_analysis_progress("Analysis completed", 100)
            
            logger.info("Embedding space analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing embedding space: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __del__(self):
        """Clean up resources when the interpreter is destroyed."""
        try:
            if self.worker_id is not None:
                from simpler_fine_bert.tokenizer_manager import tokenizer_manager
                tokenizer_manager.cleanup_worker(self.worker_id)
                logger.info(f"Cleaned up tokenizer resources for worker {self.worker_id}")
        except Exception as e:
            logger.error(f"Error cleaning up tokenizer resources: {str(e)}")

def create_interpreter(
    model: nn.Module,
    worker_id: int,
    model_name: str,
    device: torch.device,
    wandb_manager: Optional[Any] = None
) -> EmbeddingInterpreter:
    logger.info("Creating EmbeddingInterpreter")
    try:
        # Get tokenizer through manager
        from simpler_fine_bert.tokenizer_manager import tokenizer_manager
        tokenizer = tokenizer_manager.get_worker_tokenizer(
            worker_id=worker_id,
            model_name=model_name
        )
        interpreter = EmbeddingInterpreter(
            model=model,
            tokenizer=tokenizer,
            device=device,
            wandb_manager=wandb_manager,
            worker_id=worker_id
        )
        logger.info("EmbeddingInterpreter created successfully")
        return interpreter
    except Exception as e:
        logger.error(f"Error creating interpreter: {str(e)}")
        logger.error(traceback.format_exc())
        raise
