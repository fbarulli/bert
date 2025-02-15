import torch
import random
import logging
from typing import Tuple, Optional, List, Set, Dict
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
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply whole word masking to input tokens.
        
        Args:
            batch: Dictionary containing:
                - input_ids: Input token IDs [seq_len]
                - word_ids: Word IDs for tokens [seq_len]
                - special_tokens_mask: Special tokens mask [seq_len]
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']
        
        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")
            
        # Create CPU tensors
        input_ids = tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()
        
        # Get word boundaries using word IDs and special tokens mask
        word_ids_list = [None if id == -1 or mask == 1 else id for id, mask in zip(word_ids.tolist(), special_tokens_mask.tolist())]
        
        # Get word boundaries using existing word IDs
        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(word_boundaries, word_ids_list)
        
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
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply span masking to input tokens.
        
        Args:
            batch: Dictionary containing:
                - input_ids: Input token IDs [seq_len]
                - word_ids: Word IDs for tokens [seq_len]
                - special_tokens_mask: Special tokens mask [seq_len]
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        input_ids = batch['input_ids']
        word_ids = batch['word_ids']
        special_tokens_mask = batch['special_tokens_mask']
        
        if input_ids.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {input_ids.shape}")
            
        # Create CPU tensors
        input_ids = tensor_manager.create_cpu_tensor(input_ids.clone(), dtype=torch.long)
        original_ids = input_ids.clone()
        
        # Get word boundaries using word IDs and special tokens mask
        word_ids_list = [None if id == -1 or mask == 1 else id for id, mask in zip(word_ids.tolist(), special_tokens_mask.tolist())]
        
        # Get word boundaries using existing word IDs
        word_boundaries = self._get_word_boundaries(input_ids, word_ids_list)
        maskable_boundaries = self._get_maskable_boundaries(
            word_boundaries,
            word_ids_list,
            self.max_span_length
        )
        
        if not maskable_boundaries:
            logger.warning("No maskable word boundaries found")
            return input_ids, torch.full_like(input_ids, -100)
        
        # Calculate target based on total sequence length
        seq_length = len(input_ids)
        target_masked = int(seq_length * self.mask_prob)  # Target 15% of total sequence
        
        logger.debug(
            f"Masking targets:\n"
            f"- Sequence length: {seq_length}\n"
            f"- Target tokens: {target_masked} ({self.mask_prob:.1%})\n"
            f"- Maskable boundaries: {len(maskable_boundaries)}"
        )
        
        # Analyze token overlap in boundaries
        token_counts = {}  # Track how many boundaries each token appears in
        boundary_tokens = []  # Track tokens in each boundary
        for i, (start, end) in enumerate(maskable_boundaries):
            tokens = set(range(start, end))
            boundary_tokens.append(tokens)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Calculate overlap statistics
        all_tokens = set(token_counts.keys())
        overlap_tokens = {t for t, c in token_counts.items() if c > 1}
        max_overlap = max(token_counts.values()) if token_counts else 0
        avg_overlap = sum(token_counts.values()) / len(token_counts) if token_counts else 0
        
        logger.debug(
            f"Boundary overlap analysis:\n"
            f"- Total tokens: {len(all_tokens)}\n"
            f"- Tokens in multiple boundaries: {len(overlap_tokens)}\n"
            f"- Max overlap count: {max_overlap}\n"
            f"- Average overlap: {avg_overlap:.1f}"
        )
        
        # Preprocess boundaries with overlap info
        processed_boundaries = []
        for i, (start, end) in enumerate(maskable_boundaries):
            length = end - start
            overlap_score = sum(token_counts[t] for t in range(start, end))
            processed_boundaries.append((i, start, end, length, overlap_score))
        
        # Sort by length and overlap score
        processed_boundaries.sort(key=lambda x: (x[3], x[4]), reverse=True)
        
        # Initialize masking state
        masked_positions = set()
        successful_attempts = 0
        attempts = 0
        max_attempts = len(maskable_boundaries) * 2
        
        logger.debug(
            f"Token distribution:\n"
            f"- Unique tokens: {len(all_tokens)}\n"
            f"- Tokens per boundary: {len(all_tokens)/len(maskable_boundaries):.1f}\n"
            f"- Coverage: {len(all_tokens)/seq_length:.1%}"
        )
        
        # Track phase statistics
        first_pass_tokens = 0
        second_pass_tokens = 0
        fallback_tokens = 0
        used_boundaries = set()
        
        # First pass: Try longer spans first
        for idx, start, end, length, overlap_score in processed_boundaries:
            if len(masked_positions) >= target_masked:
                break
                
            # Apply masking
            new_tokens = set(range(start, end)) - masked_positions
            if new_tokens:
                self._apply_token_masking(input_ids, start, end)
                masked_positions.update(new_tokens)
                used_boundaries.add(idx)
                first_pass_tokens += len(new_tokens)
                successful_attempts += 1
                
                logger.debug(
                    f"First pass masking:\n"
                    f"- Boundary {idx} (length {length}, overlap {overlap_score})\n"
                    f"- New tokens: {len(new_tokens)}\n"
                    f"- Current ratio: {len(masked_positions)/seq_length:.1%}"
                )
        
        # Second pass: Reuse effective boundaries if needed
        while len(masked_positions) < target_masked and attempts < max_attempts:
            attempts += 1
            
            # Try boundaries that gave most tokens first
            remaining_target = target_masked - len(masked_positions)
            for idx, start, end, length, overlap_score in processed_boundaries:
                if len(masked_positions) >= target_masked:
                    break
                    
                # Skip if this boundary wouldn't help
                new_tokens = set(range(start, end)) - masked_positions
                if len(new_tokens) == 0:
                    continue
                    
                # Apply masking
                self._apply_token_masking(input_ids, start, end)
                masked_positions.update(new_tokens)
                used_boundaries.add(idx)
                second_pass_tokens += len(new_tokens)
                successful_attempts += 1
                
                logger.debug(
                    f"Second pass masking (attempt {attempts}):\n"
                    f"- Boundary {idx} (length {length}, overlap {overlap_score})\n"
                    f"- New tokens: {len(new_tokens)}\n"
                    f"- Current ratio: {len(masked_positions)/seq_length:.1%}"
                )
        
        # Fallback: Try individual tokens if still under target
        if len(masked_positions) < target_masked:
            logger.debug("Using fallback token-level masking")
            
            # Sort tokens by overlap count
            token_items = sorted(
                token_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Try tokens in order of overlap count
            for token_idx, overlap_count in token_items:
                if token_idx in masked_positions:
                    continue
                    
                if len(masked_positions) >= target_masked:
                    break
                    
                self._apply_token_masking(input_ids, token_idx, token_idx + 1)
                masked_positions.add(token_idx)
                fallback_tokens += 1
                successful_attempts += 1
                
        # Log phase effectiveness
        total_tokens = len(masked_positions)
        if total_tokens > 0:
            logger.debug(
                f"Phase effectiveness:\n"
                f"- First pass: {first_pass_tokens} tokens ({first_pass_tokens/total_tokens:.1%})\n"
                f"- Second pass: {second_pass_tokens} tokens ({second_pass_tokens/total_tokens:.1%})\n"
                f"- Fallback: {fallback_tokens} tokens ({fallback_tokens/total_tokens:.1%})\n"
                f"- Success rate: {successful_attempts/attempts:.1%} ({successful_attempts}/{attempts})"
            )
            
        # Create labels
        labels = self._create_labels(original_ids, masked_positions)
        
        # Log final statistics
        num_masked = len(masked_positions)
        mask_ratio = num_masked / seq_length
        logger.info(
            f"Final masking results:\n"
            f"- Sequence length: {seq_length}\n"
            f"- Target tokens: {target_masked} ({self.mask_prob:.1%})\n"
            f"- Masked tokens: {num_masked}\n"
            f"- Achieved ratio: {mask_ratio:.1%}\n"
            f"- Maskable boundaries: {len(maskable_boundaries)}"
        )
        
        # Warn if masking ratio is too low
        if mask_ratio < self.MIN_MASK_PROB:
            logger.warning(
                f"Low masking ratio {mask_ratio:.1%} < {self.MIN_MASK_PROB:.1%}\n"
                f"- Target ratio: {self.mask_prob:.1%}\n"
                f"- Attempts made: {attempts}\n"
                f"- Masked tokens: {num_masked}\n"
                f"- Maskable boundaries: {len(maskable_boundaries)}"
            )
        
        return input_ids, labels

def create_attention_mask(input_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """Create attention mask from input ids."""
    return (input_ids != padding_idx).float()
