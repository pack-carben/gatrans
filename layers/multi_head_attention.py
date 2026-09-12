import math
from typing import Optional, Tuple

import torch
from torch import nn

from .spatial_encoding import SpatialEncoding


class MultiHeadAttention(nn.Module):
    """Graphormer attention using node features as queries and spatial encodings as keys/values."""

    def __init__(self, d_model: int, num_heads: int, n_neighbors: int, d_sp_enc: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.spatial_encoding = SpatialEncoding(n_neighbors, d_sp_enc, d_model)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        spatial_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_bias = self.spatial_encoding(spatial_matrix)

        q = self._split_heads(self.wq(x))
        k = self._split_heads(self.wk(spatial_bias))
        v = self._split_heads(self.wv(spatial_bias))

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask[:, None, :, :], -1e9)

        attention = torch.softmax(logits, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        return self.output(out), attention
