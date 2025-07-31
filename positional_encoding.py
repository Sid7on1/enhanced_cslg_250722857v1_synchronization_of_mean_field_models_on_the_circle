import torch
import math
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(torch.nn.Module):
    """
    Positional Encoding module that adds positional information to the input data.

    Args:
        d_model: Dimensionality of the input data.
        dropout: Dropout probability.
        max_len: Maximum sequence length.
        padding_idx: Index of the padding token. Positional encoding for this index will be all zeros.

    Attributes:
        dropout: Dropout layer.
        pos_embed: Positional encodings for each position and dimension.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, padding_idx: int = 0):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.pos_embed = self._init_positional_encoding(d_model, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encodings added, of the same shape as input.
        """
        # Add positional encodings to the input tensor
        if self.padding_idx is not None:
            # Set positional encoding for padding token to all zeros
            x = torch.where(x == self.padding_idx, x, x + self.pos_embed[:x.size(1), :].to(x.device))
        else:
            x += self.pos_embed[:x.size(1), :].to(x.device)

        # Apply dropout
        x = self.dropout(x)

        return x

    def _init_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """
        Initialize the positional encodings.

        Args:
            d_model: Dimensionality of the input data.
            max_len: Maximum sequence length.

        Returns:
            Tensor of shape (max_len, d_model) containing the positional encodings.
        """
        # Create a range of positions
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        # Compute positional encodings
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # For the padding token, set positional encoding to all zeros
        if self.padding_idx is not None:
            pe[self.padding_idx, :] = 0

        return pe

class LearnedPositionalEncoding(torch.nn.Module):
    """
    Learned Positional Encoding module that adds learnable positional embeddings to the input data.

    Args:
        embedding_dim: Dimensionality of the input data and embeddings.
        max_len: Maximum sequence length.
        padding_idx: Index of the padding token. Positional encoding for this index will be a learned embedding.

    Attributes:
        pos_embed: Learnable positional embeddings of shape (max_len, embedding_dim).
    """
    def __init__(self, embedding_dim: int, max_len: int = 5000, padding_idx: int = 0):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_embed = torch.nn.Embedding(num_embeddings=max_len, embedding_dim=embedding_dim)
        self.padding_idx = padding_idx
        self._init_embeddings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the learned positional encoding layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len).

        Returns:
            Tensor with learned positional embeddings added, of shape (batch_size, seq_len, embedding_dim).
        """
        # Add learned positional embeddings to the input tensor
        pos_embed = self.pos_embed.weight
        if self.padding_idx is not None:
            # Use learned embedding for the padding token
            x = torch.where(x == self.padding_idx, x, x + pos_embed[:x.size(1), :].to(x.device))
        else:
            x = x + pos_embed[:x.size(1), :].to(x.device)

        return x

    def _init_embeddings(self):
        """Initialize the positional embeddings."""
        # Initialize embeddings using normal distribution
        self.pos_embed.weight.data.normal_(mean=0, std=0.1)

def add_positional_encoding(x: torch.Tensor,
                           embedding_dim: int,
                           dropout: float = 0.1,
                           max_len: int = 5000,
                           padding_idx: int = 0,
                           learned: bool = False) -> torch.Tensor:
    """
    Add positional encodings to the input tensor.

    Args:
        x: Input tensor of shape (batch_size, seq_len).
        embedding_dim: Dimensionality of the input data and embeddings.
        dropout: Dropout probability.
        max_len: Maximum sequence length.
        padding_idx: Index of the padding token. Positional encoding for this index will be all zeros or a learned embedding.
        learned: Whether to use learned positional embeddings.

    Returns:
        Tensor with positional encodings added, of shape (batch_size, seq_len, embedding_dim).
    """
    # Create positional encoding module based on the learned parameter
    if learned:
        pos_encoding = LearnedPositionalEncoding(embedding_dim, max_len, padding_idx)
    else:
        pos_encoding = PositionalEncoding(embedding_dim, dropout, max_len, padding_idx)

    # Add positional encodings to the input tensor
    return pos_encoding(x)

# Example usage
if __name__ == "__main__":
    import numpy as np

    # Sample input data
    batch_size = 32
    seq_len = 100
    embedding_dim = 512
    x = torch.from_numpy(np.random.randint(low=0, high=1000, size=(batch_size, seq_len)))

    # Add positional encodings
    pos_encoded_x = add_positional_encoding(x, embedding_dim, padding_idx=0, learned=True)
    print(pos_encoded_x.shape)  # Should print: torch.Size([32, 100, 512])