import torch
from torch import nn


class CentralityEncoding(nn.Module):
    """Embedding layer for node degree inside sampled subgraphs."""

    def __init__(self, max_degree: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_degree + 1, d_model)

    def forward(self, degree: torch.Tensor) -> torch.Tensor:
        degree = degree.long().clamp(min=0, max=self.embedding.num_embeddings - 1)
        return self.embedding(degree)
