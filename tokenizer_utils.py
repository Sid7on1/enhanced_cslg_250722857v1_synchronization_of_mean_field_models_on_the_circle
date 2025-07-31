import logging
import string
import re
import unicodedata
from typing import List, Dict, Tuple, Union
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerUtils:
    """
    Tokenization utilities for transformer models.

    ...

    Attributes
    ----------
    tokenizer : BertTokenizerFast
        Hugging Face's Bert tokenizer for tokenizing input text.
    max_seq_length : int
        Maximum allowed sequence length for tokenized input.
    pad_token : str
        Padding token used for sequences shorter than max_seq_length.
    do_lower_case : bool
        Flag indicating whether to convert text to lowercase during tokenization.
    mask_token : str
        Mask token used for masked language modeling tasks.
    unk_token : str
        Unknown token used for out-of-vocabulary words.

    Methods
    -------
    tokenize(text: str) -> Dict[str, np.array]:
        Tokenize input text and return input_ids, attention_masks, and token_type_ids.

    preprocess_text(text: str) -> str:
        Preprocess text by removing punctuation, converting to lowercase, etc.

    truncate_sequence(tokens: List[str], max_seq_length: int) -> List[str]:
        Truncate token sequence to the specified maximum sequence length.

    pad_sequence(tokens: List[str], max_seq_length: int) -> np.array:
        Pad token sequence to the specified maximum sequence length.

    tokenize_batch(texts: List[str]) -> Dict[str, np.array]:
        Tokenize a batch of input texts.

    """

    def __init__(self,
                 max_seq_length: int = 512,
                 do_lower_case: bool = True,
                 mask_token: str = "[MASK]",
                 pad_token: str = "[PAD]",
                 unk_token: str = "[UNK]"):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') if do_lower_case else BertTokenizerFast.from_pretrained(
            'bert-base-cased')
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.do_lower_case = do_lower_case
        self.mask_token = mask_token
        self.unk_token = unk_token

    def tokenize(self, text: str) -> Dict[str, np.array]:
        """
        Tokenize input text and return input IDs, attention masks, and token type IDs.

        Parameters
        ----------
        text : str
            Input text to be tokenized.

        Returns
        -------
        Dict[str, np.array]
            Dictionary containing input_ids, attention_masks, and token_type_ids.

        """
        tokens = self.tokenizer.tokenize(text)
        tokens = self.truncate_sequence(tokens, self.max_seq_length)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens = np.array(tokens, dtype=np.int32)

        input_ids = tokens
        attention_masks = [1] * len(tokens)
        token_type_ids = [0] * len(tokens)

        input_ids = np.array(input_ids, dtype=np.int32)
        attention_masks = np.array(attention_masks, dtype=np.int32)
        token_type_ids = np.array(token_type_ids, dtype=np.int32)

        return {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "token_type_ids": token_type_ids
        }

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing punctuation, converting to lowercase, etc.

        Parameters
        ----------
        text : str
            Input text to be preprocessed.

        Returns
        -------
        str
            Preprocessed text.

        """
        text = self._remove_punctuation(text)
        text = self._strip_accents(text)
        text = text.lower() if self.do_lower_case else text
        return text

    def truncate_sequence(self, tokens: List[str], max_seq_length: int) -> List[str]:
        """
        Truncate token sequence to the specified maximum sequence length.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to be truncated.
        max_seq_length : int
            Maximum allowed sequence length.

        Returns
        -------
        List[str]
            Truncated list of tokens.

        """
        while len(tokens) > max_seq_length:
            tokens.pop()
        return tokens

    def pad_sequence(self, tokens: List[str], max_seq_length: int) -> np.array:
        """
        Pad token sequence to the specified maximum sequence length.

        Parameters
        ----------
        tokens : List[str]
            List of tokens to be padded.
        max_seq_length : int
            Maximum allowed sequence length.

        Returns
        -------
        np.array
            Padded token IDs with shape (max_seq_length,).

        """
        if len(tokens) < max_seq_length:
            num_pads = max_seq_length - len(tokens)
            padding = [self.tokenizer.pad_token_id] * num_pads
            tokens += padding
        tokens = tokens[:max_seq_length]
        tokens = np.array(tokens, dtype=np.int32)
        return tokens

    def tokenize_batch(self, texts: List[str]) -> Dict[str, np.array]:
        """
        Tokenize a batch of input texts.

        Parameters
        ----------
        texts : List[str]
            List of input texts to be tokenized.

        Returns
        -------
        Dict[str, np.array]
            Dictionary containing input_ids, attention_masks, and token_type_ids for the batch.

        """
        input_ids_batch = []
        attention_masks_batch = []
        token_type_ids_batch = []

        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            tokens = self.truncate_sequence(tokens, self.max_seq_length - 2)
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            tokens = self.pad_sequence(tokens, self.max_seq_length)

            input_ids = tokens
            attention_masks = [1] * len(tokens)
            token_type_ids = [0] * len(tokens)

            input_ids_batch.append(input_ids)
            attention_masks_batch.append(attention_masks)
            token_type_ids_batch.append(token_type_ids)

        input_ids_batch = np.array(input_ids_batch, dtype=np.int32)
        attention_masks_batch = np.array(attention_masks_batch, dtype=np.int32)
        token_type_ids_batch = np.array(token_type_ids_batch, dtype=np.int32)

        return {
            "input_ids": input_ids_batch,
            "attention_masks": attention_masks_batch,
            "token_type_ids": token_type_ids_batch
        }

    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from the input text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with punctuation removed.

        """
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def _strip_accents(self, text: str) -> str:
        """
        Remove accents from Unicode characters in the input text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with accents removed.

        """
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')

# Example usage
if __name__ == "__main__":
    tokenizer_utils = TokenizerUtils()
    text = "This is a sample text for tokenization."
    preprocessed_text = tokenizer_utils.preprocess_text(text)
    tokens = tokenizer_utils.tokenize(preprocessed_text)
    print("Preprocessed Text:", preprocessed_text)
    print("Tokens:", tokens)

    batch_texts = ["This is the first sentence.", "This is another sentence."]
    batch_tokens = tokenizer_utils.tokenize_batch(batch_texts)
    print("Batch Tokens:", batch_tokens)