import torch
from torch import nn


class SpatialEncoding(nn.Module):
    """Project shortest-path distances in a sampled subgraph to attention keys."""

    def __init__(self, n_neighbors: int, d_sp_enc: int, d_model: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_neighbors, d_sp_enc),
            nn.ReLU(),
            nn.Linear(d_sp_enc, d_model),
            nn.ReLU(),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.float()
        distances = torch.where(distances < 0, torch.zeros_like(distances), distances)
        return self.network(distances)
