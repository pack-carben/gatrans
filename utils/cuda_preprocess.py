"""CUDA random walks and exact full-network shortest-path extraction."""

import time

import numpy as np
import torch
from tqdm.auto import tqdm


def _transpose_csr(edge_sources, edge_targets, n_nodes):
    """Build A.T in CSR form so one sparse matmul advances many BFS fronts."""
    order = torch.argsort(edge_targets * n_nodes + edge_sources)
    rows = edge_targets[order]
    columns = edge_sources[order]
    counts = torch.bincount(rows, minlength=n_nodes)
    crow = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    values = torch.ones(columns.numel(), dtype=torch.float32, device=columns.device)
    return torch.sparse_csr_tensor(
        crow, columns, values, size=(n_nodes, n_nodes), device=columns.device
    )


def sampled_global_distances(edge_sources, edge_targets, n_nodes, walks, source_batch_size=512):
    """Return full-graph shortest paths for every node pair in each sampled walk.

    Distances are computed by batched multi-source BFS over the complete graph.
    Only the sampled pairs are retained, so no dense N x N matrix is stored.
    """
    if source_batch_size < 1:
        raise ValueError("source_batch_size must be positive.")
    if walks.ndim < 2:
        raise ValueError("walks must have at least two dimensions.")
    device = walks.device
    edge_sources = edge_sources.to(device=device, dtype=torch.long)
    edge_targets = edge_targets.to(device=device, dtype=torch.long)
    graph_t = _transpose_csr(edge_sources, edge_targets, n_nodes)

    walk_size = walks.shape[-1]
    walk_rows = walks.reshape(-1, walk_size).long()
    flat_sources = walk_rows.reshape(-1)
    occurrence_order = torch.argsort(flat_sources)
    occurrence_counts = torch.bincount(flat_sources, minlength=n_nodes)
    occurrence_offsets = torch.cat(
        (occurrence_counts.new_zeros(1), occurrence_counts.cumsum(0))
    )
    result = torch.empty((flat_sources.numel(), walk_size), dtype=torch.int32, device=device)

    for start in tqdm(
        range(0, n_nodes, source_batch_size),
        desc="CUDA full-network shortest paths",
        unit="source-batch",
    ):
        end = min(start + source_batch_size, n_nodes)
        width = end - start
        source_ids = torch.arange(start, end, device=device)
        batch_ids = torch.arange(width, device=device)

        frontier = torch.zeros((n_nodes, width), dtype=torch.float32, device=device)
        frontier[source_ids, batch_ids] = 1
        visited = frontier.bool()
        distance = torch.full((n_nodes, width), -1, dtype=torch.int32, device=device)
        distance[source_ids, batch_ids] = 0

        for depth in range(1, n_nodes):
            reached = torch.sparse.mm(graph_t, frontier) > 0
            reached.logical_and_(~visited)
            if not torch.any(reached):
                break
            distance.masked_fill_(reached, depth)
            visited.logical_or_(reached)
            frontier = reached.float()

        left = int(occurrence_offsets[start].item())
        right = int(occurrence_offsets[end].item())
        occurrences = occurrence_order[left:right]
        occurrence_sources = flat_sources[occurrences] - start
        sampled_rows = torch.div(occurrences, walk_size, rounding_mode="floor")
        sampled_targets = walk_rows[sampled_rows]
        result[occurrences] = distance[sampled_targets, occurrence_sources[:, None]]

    return result.reshape(*walks.shape, walk_size).float()


def sampled_distances(adjacency, walks, source_batch_size=512):
    """Compatibility wrapper for exact full-network sampled-pair distances."""
    sources, targets = adjacency.nonzero(as_tuple=True)
    return sampled_global_distances(
        sources, targets, adjacency.shape[0], walks, source_batch_size
    )


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
    edge_sources, edge_targets = adjacency.nonzero(as_tuple=True)
    n_nodes = adjacency.shape[0]
    print(f"CUDA graph: nodes={n_nodes}, edge_rows={edge_targets.numel()}, "
          f"isolated={(degrees == 0).sum().item()}; source_batch={batch_size}", flush=True)

    node_neighbor = np.empty((n_nodes, n_graphs, n_neighbors), dtype=np.int64)
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in tqdm(range(0, n_nodes, batch_size), desc="CUDA random walks", unit="node-batch"):
        end = min(start + batch_size, n_nodes)
        current = torch.arange(start, end, device=device).repeat_interleave(n_graphs)
        walks = torch.empty((current.numel(), n_neighbors), dtype=torch.long, device=device)
        walks[:, 0] = current
        for step in range(1, n_neighbors):
            counts = degrees[current]
            if edge_targets.numel():
                choice = (torch.rand(current.shape, device=device, generator=generator) * counts).long()
                positions = (offsets[current] + choice).clamp(max=edge_targets.numel() - 1)
                current = torch.where(counts > 0, edge_targets[positions], current)
            walks[:, step] = current
        node_neighbor[start:end] = walks.reshape(end - start, n_graphs, n_neighbors).cpu().numpy()

    all_walks = torch.as_tensor(node_neighbor, dtype=torch.long, device=device)
    del adjacency
    torch.cuda.empty_cache()
    spatial_matrix = sampled_global_distances(
        edge_sources, edge_targets, n_nodes, all_walks, source_batch_size=batch_size
    ).cpu().numpy()
    print(f"CUDA preprocessing completed in {time.perf_counter() - started:.2f}s", flush=True)
    return node_neighbor, spatial_matrix, degrees.cpu().numpy()
