from typing import Tuple

import torch
from torch import nn


class ShapFeatureWrapper(nn.Module):
    """Expose TREE as a dense-feature model for SHAP explainers.

    The production model receives integer node ids. SHAP works better on
    differentiable tensors, so this wrapper accepts a flattened subgraph tensor
    and reconstructs node features, degrees, and spatial matrices before calling
    TREE.forward_features().
    """

    def __init__(self, model, n_graphs: int, n_neighbors: int, feature_dim: int):
        super().__init__()
        self.model = model
        self.n_graphs = n_graphs
        self.n_neighbors = n_neighbors
        self.feature_dim = feature_dim

        self.feature_size = n_graphs * n_neighbors * feature_dim
        self.degree_size = n_graphs * n_neighbors
        self.spatial_size = n_graphs * n_neighbors * n_neighbors

    def split(self, flat_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_end = self.feature_size
        degree_end = feature_end + self.degree_size

        node_feature = flat_feature[:, :feature_end].view(
            -1, self.n_graphs, self.n_neighbors, self.feature_dim
        )
        degree = flat_feature[:, feature_end:degree_end].view(
            -1, self.n_graphs, self.n_neighbors
        )
        spatial = flat_feature[:, degree_end : degree_end + self.spatial_size].view(
            -1, self.n_graphs, self.n_neighbors, self.n_neighbors
        )
        return node_feature, degree, spatial

    def forward(self, flat_feature: torch.Tensor) -> torch.Tensor:
        node_feature, degree, spatial = self.split(flat_feature)
        embedding, _ = self.model.forward_features(node_feature, degree.long(), spatial)
        return torch.sigmoid(self.model.classifier(embedding))
