from __future__ import annotations

import logging
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from typing import Dict, Any, Optional, Tuple, Union
from simpler_fine_bert.common.managers import (
    get_cuda_manager,
    get_batch_manager,
    get_tensor_manager
)

# Get manager instances
cuda_manager = get_cuda_manager()
batch_manager = get_batch_manager()
tensor_manager = get_tensor_manager()

logger = logging.getLogger(__name__)

class ClassificationBert(BertPreTrainedModel):
    """BERT model for classification tasks."""
    
    def __init__(
        self,
        config: BertConfig,
        num_labels: int = 2,
        dropout_prob: float = 0.1,
        hidden_size: Optional[int] = None
    ):
        """Initialize model.
        
        Args:
            config: Model configuration
            num_labels: Number of output labels
            dropout_prob: Dropout probability
            hidden_size: Optional hidden layer size
        """
        super().__init__(config)
        
        # Initialize BERT
        self.bert = BertModel(config)
        
        # Get dimensions
        bert_hidden_size = config.hidden_size
        if hidden_size is None:
            hidden_size = bert_hidden_size
        
        # Create classifier
        self.dropout = nn.Dropout(dropout_prob)
        if hidden_size != bert_hidden_size:
            self.intermediate = nn.Linear(bert_hidden_size, hidden_size)
            self.activation = nn.GELU()
        else:
            self.intermediate = None
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights(self.classifier)
        if self.intermediate is not None:
            self._init_weights(self.intermediate)
        
        logger.info(
            f"Initialized ClassificationBert with:\n"
            f"- BERT hidden size: {bert_hidden_size}\n"
            f"- Classifier hidden size: {hidden_size}\n"
            f"- Number of labels: {num_labels}\n"
            f"- Dropout probability: {dropout_prob}"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Optional token type IDs
            position_ids: Optional position IDs
            labels: Optional labels for loss calculation
            output_hidden_states: Whether to output all hidden states
            output_attentions: Whether to output attention weights
            return_dict: Whether to return dict or tuple
            
        Returns:
            Model outputs
        """
        # Move inputs to correct device using batch_manager
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        if position_ids is not None:
            inputs['position_ids'] = position_ids
        if labels is not None:
            inputs['labels'] = labels
            
        inputs = batch_manager.prepare_batch(inputs, self.device)
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids'),
            position_ids=inputs.get('position_ids'),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict
        )
        
        # Get pooled output
        pooled_output = outputs[1]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Apply intermediate layer if present
        if self.intermediate is not None:
            pooled_output = self.intermediate(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if 'labels' in inputs:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), inputs['labels'].view(-1))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': outputs.hidden_states if output_hidden_states else None,
                'attentions': outputs.attentions if output_attentions else None
            }
        
        # Return tuple for backwards compatibility
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
