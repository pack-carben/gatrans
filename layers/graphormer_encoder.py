from typing import Optional, Tuple

import torch
from torch import nn

from .multi_head_attention import MultiHeadAttention


class GraphormerBlock(nn.Module):
    """Pre-norm Graphormer block used for each sampled random-walk channel."""

    def __init__(self, d_model: int, num_heads: int, dff: int, n_neighbors: int, d_sp_enc: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, n_neighbors, d_sp_enc)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        spatial_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attention = self.attention(self.norm1(x), spatial_matrix, attention_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attention
