"""Batched random walks and induced-subgraph shortest paths on CUDA."""

import time

import numpy as np
import torch
from tqdm.auto import tqdm


def sampled_distances(adjacency, walks):
    """Floyd-Warshall over each sampled graph, including repeated node IDs."""
    size = walks.shape[-1]
    edges = adjacency[walks[:, :, None], walks[:, None, :]]
    distances = torch.where(edges, 1, size + 1)
    distances.masked_fill_(walks[:, :, None] == walks[:, None, :], 0)
    for k in range(size):
        distances = torch.minimum(distances, distances[:, :, k:k + 1] + distances[:, k:k + 1, :])
    return distances.masked_fill(distances > size, -1).float()


def build_subgraphs_cuda(adj, n_graphs, n_neighbors, seed=42, batch_size=512):
    if not torch.cuda.is_available():
        raise RuntimeError("--engine cuda requires a CUDA-enabled PyTorch and an available GPU.")
    if min(n_graphs, n_neighbors, batch_size) < 1:
        raise ValueError("n_graphs, n_neighbors and batch_size must be positive.")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Expected a square adjacency matrix.")
    started = time.perf_counter()
    device = torch.device("cuda")
    print(f"Uploading adjacency to {torch.cuda.get_device_name(device)}...", flush=True)
    adjacency = torch.as_tensor(np.asarray(adj) > 0, device=device)
    degrees = adjacency.sum(dim=1)
    offsets = torch.cat([degrees.new_zeros(1), degrees.cumsum(0)])
    # torch.nonzero returns row-major indices, so offsets address each row's neighbors.
    targets = adjacency.nonzero(as_tuple=True)[1]
    n_nodes = adjacency.shape[0]
    print(f"CUDA graph: nodes={n_nodes}, edge_rows={targets.numel()}, "
          f"isolated={(degrees == 0).sum().item()}; batch_nodes={batch_size}", flush=True)
    node_neighbor = np.empty((n_nodes, n_graphs, n_neighbors), dtype=np.int64)
    spatial_matrix = np.empty((n_nodes, n_graphs, n_neighbors, n_neighbors), dtype=np.float32)
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in tqdm(range(0, n_nodes, batch_size), desc="CUDA sampling + shortest paths", unit="batch"):
        end = min(start + batch_size, n_nodes)
        current = torch.arange(start, end, device=device).repeat_interleave(n_graphs)
        walks = torch.empty((current.numel(), n_neighbors), dtype=torch.long, device=device)
        walks[:, 0] = current
        for step in range(1, n_neighbors):
            if targets.numel():
                counts = degrees[current]
                choice = (torch.rand(current.shape, device=device, generator=generator) * counts).long()
                positions = (offsets[current] + choice).clamp(max=targets.numel() - 1)
                current = torch.where(counts > 0, targets[positions], current)
            walks[:, step] = current
        distances = sampled_distances(adjacency, walks)
        node_neighbor[start:end] = walks.reshape(end - start, n_graphs, n_neighbors).cpu().numpy()
        spatial_matrix[start:end] = distances.reshape(end - start, n_graphs, n_neighbors, n_neighbors).cpu().numpy()
    print(f"CUDA preprocessing completed in {time.perf_counter() - started:.2f}s", flush=True)
    return node_neighbor, spatial_matrix, degrees.cpu().numpy()
