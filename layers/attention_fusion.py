import math
from typing import Tuple

import torch
from torch import nn


class AttentionFusion(nn.Module):
    """Fuse target-node embeddings from multiple sampled graph channels."""

    def __init__(self, d_model: int, n_channels: int):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, channel_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.wq(channel_embeddings)
        k = self.wk(channel_embeddings)
        v = self.wv(channel_embeddings)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        weights = torch.softmax(logits.mean(dim=1, keepdim=True), dim=-1)
        fused = torch.matmul(weights, v).squeeze(1)
        return self.norm(fused), weights.squeeze(1)
