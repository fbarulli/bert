import torch
import random
import logging
from typing import Tuple, Optional, List, Set
from transformers import PreTrainedTokenizerFast

from simpler_fine_bert.common.managers import (
    get_tensor_manager,
    get_tokenizer_manager
)

# Get manager instances
tensor_manager = get_tensor_manager()
tokenizer_manager = get_tokenizer_manager()

logger = logging.getLogger(__name__)

class MaskingModule:
    """Base class for masking strategies used in embedding learning."""
    
    # Hyperparameter ranges from config
    MIN_MASK_PROB = 0.1
    MAX_MASK_PROB = 0.3
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        mask_prob: float = 0.15,
        max_predictions: int = 20,
        worker_id: Optional[int] = None
    ):
        """Initialize masking module.
        
        Args:
            tokenizer: BERT tokenizer
            mask_prob: Probability of masking each word (between 0.1 and 0.3)
            max_predictions: Maximum number of tokens to mask
            worker_id: Optional worker ID for process-specific resources
        """
        # Validate mask probability
        if not self.MIN_MASK_PROB <= mask_prob <= self.MAX_MASK_PROB:
            logger.warning(
                f"Mask probability {mask_prob} outside recommended range "
                f"[{self.MIN_MASK_PROB}, {self.MAX_MASK_PROB}]"
            )
        self.mask_prob = max(self.MIN_MASK_PROB, min(self.MAX_MASK_PROB, mask_prob))
        
        self.tokenizer = tokenizer
        self.base_max_predictions = max_predictions
        self.max_predictions = max_predictions
        
        # Get special token IDs
        self.special_token_ids = set([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id
        ])
        
        # Get valid vocabulary for random token selection
        self.valid_vocab_ids = [
            i for i in range(tokenizer.vocab_size)
            if i not in self.special_token_ids
        ]
        
        logger.debug(
            f"Initialized masking module:\n"
            f"- Mask probability: {self.mask_prob:.2%}\n"
            f"- Base max predictions: {self.base_max_predictions}\n"
            f"- Special tokens: {len(self.special_token_ids)}\n"
            f"- Valid vocab size: {len(self.valid_vocab_ids)}"
        )
    
    def _get_word_boundaries(
        self,
        input_ids: torch.Tensor,
        word_ids: List[Optional[int]]
    ) -> List[Tuple[int, int]]:
        """Get word boundaries respecting word pieces.
        
        Args:
            input_ids: Input token IDs
            word_ids: Word IDs from tokenizer
            
        Returns:
            List of (start, end) tuples for each word
        """
        word_boundaries = []
        current_word_id = None
        start_idx = None
        
        for i, word_id in enumerate(word_ids):
            # Skip special tokens
            if word_id is None:
                if start_idx is not None:
                    word_boundaries.append((start_idx, i))
                    start_idx = None
                current_word_id = None
                continue
            
            # Handle word boundaries
            if word_id != current_word_id:
                if start_idx is not None:
                    word_boundaries.append((start_idx, i))
                start_idx = i
                current_word_id = word_id
        
        # Handle final word
        if start_idx is not None:
            word_boundaries.append((start_idx, len(word_ids)))
            
        return word_boundaries
    
    def _get_maskable_boundaries(
        self,
        word_boundaries: List[Tuple[int, int]],
        word_ids: List[Optional[int]],
        max_span_length: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Get maskable word boundaries excluding special tokens.
        
        Args:
            word_boundaries: List of word boundary tuples
            word_ids: Word IDs from tokenizer
            max_span_length: Optional maximum span length for span masking
            
        Returns:
            List of maskable word boundary tuples
        """
        maskable = []
        
        for i, (start, end) in enumerate(word_boundaries):
            # Skip if any token in span is special
            if any(word_ids[j] is None for j in range(start, end)):
                continue
                
            # For span masking, ensure enough tokens remain
            if max_span_length and i + max_span_length > len(word_boundaries):
                continue
                
            maskable.append((start, end))
            
        return maskable
    
    def _apply_token_masking(
        self,
        input_ids: torch.Tensor,
        start_idx: int,
        end_idx: int
    ) -> None:
        """Apply token-level masking following BERT paper.
        
        Args:
            input_ids: Input token IDs to modify
            start_idx: Start index of span to mask
            end_idx: End index of span to mask
        """
        for idx in range(start_idx, end_idx):
            prob = random.random()
            if prob < 0.8:  # 80% mask token
                input_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:  # 10% random token (excluding special tokens)
                input_ids[idx] = random.choice(self.valid_vocab_ids)
            # 10% unchanged
    
    def _create_labels(
        self,
        original_ids: torch.Tensor,
        masked_positions: Set[int]
    ) -> torch.Tensor:
        """Create masking labels with -100 for non-masked positions.
        
        Args:
            original_ids: Original input IDs
            masked_positions: Set of positions that were selected for masking
            
        Returns:
            Label tensor with original IDs at masked positions and -100 elsewhere
        """
        labels = torch.full_like(original_ids, -100)
        for pos in masked_positions:
            labels[pos] = original_ids[pos]
        return labels

class WholeWordMaskingModule(MaskingModule):
    """Whole word masking following BERT paper."""
    
    def __call__(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply whole word masking to input tokens.
        
        Args:
            input_ids: Input token IDs [seq_len]
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")
            
        # Create CPU tensors
        input_ids = tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()
        
        # Get word IDs from tokenizer
        text = self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        word_ids = encoding.word_ids()
        
        if word_ids is None:
            raise RuntimeError("Failed to get word IDs from tokenizer")
            
        # Get word boundaries
        word_boundaries = self._get_word_boundaries(input_ids, word_ids)
        maskable_boundaries = self._get_maskable_boundaries(word_boundaries, word_ids)
        
        if not maskable_boundaries:
            logger.warning("No maskable word boundaries found")
            return input_ids, torch.full_like(input_ids, -100)
            
        # Calculate masking targets
        num_maskable_words = len(maskable_boundaries)
        num_words_to_mask = max(1, int(num_maskable_words * self.mask_prob))
        
        # Select words to mask
        words_to_mask = random.sample(maskable_boundaries, num_words_to_mask)
        masked_positions = set()
        
        # Apply masking
        for start, end in words_to_mask:
            self._apply_token_masking(input_ids, start, end)
            masked_positions.update(range(start, end))
            
        # Create labels
        labels = self._create_labels(original_ids, masked_positions)
        
        # Log statistics
        num_masked = len(masked_positions)
        logger.debug(
            f"Masking stats:\n"
            f"- Total words: {num_maskable_words}\n"
            f"- Masked words: {num_words_to_mask}\n"
            f"- Masked tokens: {num_masked}\n"
            f"- Mask ratio: {num_masked/len(input_ids):.2%}"
        )
        
        return input_ids, labels

class SpanMaskingModule(MaskingModule):
    """Span-based masking following SpanBERT paper."""
    
    MIN_SPAN_LENGTH = 1
    MAX_SPAN_LENGTH = 10
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        mask_prob: float = 0.15,
        max_span_length: int = 3,
        max_predictions: int = 20,
        worker_id: Optional[int] = None
    ):
        """Initialize span masking module.
        
        Args:
            tokenizer: BERT tokenizer
            mask_prob: Probability of masking each span
            max_span_length: Maximum length of each masked span
            max_predictions: Maximum number of tokens to mask
            worker_id: Optional worker ID for process-specific resources
        """
        super().__init__(tokenizer, mask_prob, max_predictions, worker_id)
        
        # Validate span length
        if not self.MIN_SPAN_LENGTH <= max_span_length <= self.MAX_SPAN_LENGTH:
            logger.warning(
                f"Max span length {max_span_length} outside recommended range "
                f"[{self.MIN_SPAN_LENGTH}, {self.MAX_SPAN_LENGTH}]"
            )
        self.max_span_length = max(
            self.MIN_SPAN_LENGTH,
            min(self.MAX_SPAN_LENGTH, max_span_length)
        )
        
        logger.debug(f"Using max span length: {self.max_span_length}")
    
    def __call__(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply span masking to input tokens.
        
        Args:
            input_ids: Input token IDs [seq_len]
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")
            
        # Create CPU tensors
        input_ids = tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()
        
        # Get word IDs from tokenizer
        text = self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        word_ids = encoding.word_ids()
        
        if word_ids is None:
            raise RuntimeError("Failed to get word IDs from tokenizer")
            
        # Get word boundaries
        word_boundaries = self._get_word_boundaries(input_ids, word_ids)
        maskable_boundaries = self._get_maskable_boundaries(
            word_boundaries,
            word_ids,
            self.max_span_length
        )
        
        if not maskable_boundaries:
            logger.warning("No maskable word boundaries found")
            return input_ids, torch.full_like(input_ids, -100)
            
        # Calculate number of spans needed
        total_tokens = sum(end - start for start, end in maskable_boundaries)
        target_masked = int(total_tokens * self.mask_prob)
        num_spans = max(1, (target_masked + self.max_span_length - 1) // self.max_span_length)
        
        # Select span positions
        span_positions = []
        available_positions = list(range(len(maskable_boundaries)))
        random.shuffle(available_positions)
        
        for pos in available_positions:
            if len(span_positions) >= num_spans:
                break
                
            # Ensure position has enough following words for a span
            if pos + self.max_span_length <= len(maskable_boundaries):
                span_positions.append(pos)
        
        if not span_positions:
            logger.warning("No valid span positions found")
            return input_ids, torch.full_like(input_ids, -100)
            
        # Sort positions to maintain sequence order
        span_positions.sort()
        masked_positions = set()
        
        # Apply masking for each span
        for pos in span_positions:
            # Calculate span length
            remaining = target_masked - len(masked_positions)
            if remaining <= 0:
                break
                
            max_length = min(
                self.max_span_length,
                len(maskable_boundaries) - pos,
                remaining
            )
            span_length = random.randint(1, max_length)
            
            # Get span boundaries
            span_start = maskable_boundaries[pos][0]
            span_end = maskable_boundaries[pos + span_length - 1][1]
            
            # Apply masking
            self._apply_token_masking(input_ids, span_start, span_end)
            masked_positions.update(range(span_start, span_end))
        
        # Create labels
        labels = self._create_labels(original_ids, masked_positions)
        
        # Log statistics
        num_masked = len(masked_positions)
        logger.debug(
            f"Span masking stats:\n"
            f"- Total tokens: {total_tokens}\n"
            f"- Target masked: {target_masked}\n"
            f"- Actually masked: {num_masked}\n"
            f"- Number of spans: {len(span_positions)}\n"
            f"- Mask ratio: {num_masked/len(input_ids):.2%}"
        )
        
        return input_ids, labels

def create_attention_mask(input_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """Create attention mask from input ids."""
    return (input_ids != padding_idx).float()
