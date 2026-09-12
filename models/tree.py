from typing import List, Optional, Tuple

import torch
from torch import nn

from layers import AttentionFusion, CentralityEncoding, GraphormerBlock


class TREE(nn.Module):
    """TREE-style classifier implemented with PyTorch tensors."""

    def __init__(
        self,
        node_feature,
        node_degree,
        node_neighbor,
        spatial_matrix,
        n_graphs: int,
        n_neighbors: int,
        d_model: int = 64,
        n_layers: int = 3,
        num_heads: int = 4,
        dff: int = 128,
        d_sp_enc: int = 128,
        dropout: float = 0.5,
        max_degree: Optional[int] = None,
    ):
        super().__init__()
        self.n_graphs = n_graphs
        self.n_neighbors = n_neighbors
        self.d_model = d_model

        self.register_buffer("node_feature", torch.as_tensor(node_feature, dtype=torch.float32))
        self.register_buffer("node_degree", torch.as_tensor(node_degree, dtype=torch.long))
        self.register_buffer("node_neighbor", torch.as_tensor(node_neighbor, dtype=torch.long))
        self.register_buffer("spatial_matrix", torch.as_tensor(spatial_matrix, dtype=torch.float32))

        feature_dim = self.node_feature.shape[-1]
        max_degree = int(max_degree or self.node_degree.max().item() + 1)

        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.centrality_encoding = CentralityEncoding(max_degree, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                GraphormerBlock(d_model, num_heads, dff, n_neighbors, d_sp_enc, dropout)
                for _ in range(n_layers)
            ]
        )
        self.fusion = AttentionFusion(d_model, n_graphs)
        self.classifier = nn.Linear(d_model, 1)

    def _encode_channels(
        self,
        sub_node_feature: torch.Tensor,
        sub_degree: torch.Tensor,
        sub_spatial: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        channel_outputs = []
        attentions = []

        for graph_id in range(self.n_graphs):
            x = self.feature_projection(sub_node_feature[:, graph_id])
            x = x * (self.d_model ** 0.5)
            x = x + self.centrality_encoding(sub_degree[:, graph_id])
            x = self.input_dropout(x)

            spatial = sub_spatial[:, graph_id]
            attention_mask = spatial < 0
            for block in self.blocks:
                x, attention = block(x, spatial, attention_mask)
                attentions.append(attention)

            channel_outputs.append(x[:, 0, :])

        return torch.stack(channel_outputs, dim=1), attentions

    def forward_features(
        self,
        sub_node_feature: torch.Tensor,
        sub_degree: torch.Tensor,
        sub_spatial: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        channel_embeddings, _ = self._encode_channels(sub_node_feature, sub_degree, sub_spatial)
        embedding, fusion_weights = self.fusion(channel_embeddings)
        return embedding, fusion_weights

    def forward(self, node_id: torch.Tensor) -> torch.Tensor:
        node_id = node_id.long().view(-1)
        neighbors = self.node_neighbor[node_id]
        sub_node_feature = self.node_feature[neighbors]
        sub_degree = self.node_degree[neighbors]
        sub_spatial = self.spatial_matrix[node_id]
        embedding, _ = self.forward_features(sub_node_feature, sub_degree, sub_spatial)
        return self.classifier(embedding).squeeze(-1)
