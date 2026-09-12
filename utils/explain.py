import numpy as np
import torch
from typing import List


def build_shap_features(model, node_ids: np.ndarray, device: torch.device) -> torch.Tensor:
    """Flatten sampled subgraph tensors for SHAP input."""
    ids = torch.as_tensor(node_ids, dtype=torch.long, device=device)
    neighbors = model.node_neighbor[ids]
    node_feature = model.node_feature[neighbors].float()
    degree = model.node_degree[neighbors].float().unsqueeze(-1)
    spatial = model.spatial_matrix[ids].float()

    flat = torch.cat(
        [
            node_feature.reshape(ids.numel(), -1),
            degree.reshape(ids.numel(), -1),
            spatial.reshape(ids.numel(), -1),
        ],
        dim=1,
    )
    return flat


def shap_feature_names(feature_names, n_graphs: int, n_neighbors: int) -> List[str]:
    decoded = []
    feature_names = [str(name.decode() if hasattr(name, "decode") else name) for name in feature_names]
    for graph_id in range(n_graphs):
        for neighbor_id in range(n_neighbors):
            for feature_name in feature_names:
                decoded.append(f"graph{graph_id}_node{neighbor_id}_{feature_name}")
    for graph_id in range(n_graphs):
        for neighbor_id in range(n_neighbors):
            decoded.append(f"graph{graph_id}_node{neighbor_id}_degree")
    for graph_id in range(n_graphs):
        for source_id in range(n_neighbors):
            for target_id in range(n_neighbors):
                decoded.append(f"graph{graph_id}_spatial_{source_id}_{target_id}")
    return decoded
