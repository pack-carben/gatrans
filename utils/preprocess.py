import random
import time
from collections import deque
from typing import List, Tuple

import numpy as np
from scipy import sparse
from tqdm import tqdm


def adjacency_to_neighbors(adj: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Convert a dense adjacency matrix to neighbor lists and node degrees."""
    graph = sparse.csr_matrix(adj > 0)
    degrees = np.diff(graph.indptr).astype(np.int64)
    neighbors = [graph.indices[graph.indptr[i] : graph.indptr[i + 1]] for i in range(graph.shape[0])]
    return neighbors, degrees


def random_walk(start: int, neighbors: List[np.ndarray], walk_length: int) -> List[int]:
    walk = [start]
    while len(walk) < walk_length:
        current_neighbors = neighbors[walk[-1]]
        if len(current_neighbors) == 0:
            walk.append(walk[-1])
        else:
            walk.append(int(random.choice(current_neighbors)))
    return walk


def shortest_path_in_graph(seed_nodes: np.ndarray, neighbors: List[np.ndarray]) -> np.ndarray:
    """Extract exact full-network distances for pairs in one sampled walk."""
    index = {}
    for pos, node in enumerate(seed_nodes):
        index.setdefault(int(node), []).append(pos)
    size = len(seed_nodes)
    distance = np.full((size, size), -1, dtype=np.float32)

    for source_node, source_positions in index.items():
        source_pos = source_positions[0]
        queue = deque([(int(source_node), 0)])
        seen = {int(source_node)}
        distance[source_pos, index[int(source_node)]] = 0
        remaining = set(index) - {int(source_node)}

        while queue and remaining:
            node, depth = queue.popleft()
            for nxt in neighbors[node]:
                nxt = int(nxt)
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, depth + 1))
                if nxt in remaining:
                    distance[source_pos, index[nxt]] = depth + 1
                    remaining.remove(nxt)
        distance[source_positions] = distance[source_pos]

    return distance


def shortest_path_in_subgraph(seed_nodes: np.ndarray, neighbors: List[np.ndarray]) -> np.ndarray:
    """Backward-compatible name; distances now traverse the complete graph."""
    return shortest_path_in_graph(seed_nodes, neighbors)


def build_subgraphs(
    adj: np.ndarray,
    n_graphs: int,
    n_neighbors: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random-walk channels and per-channel spatial distance matrices."""
    random.seed(seed)
    if n_graphs < 1 or n_neighbors < 1:
        raise ValueError("n_graphs and n_neighbors must be positive.")
    started = time.perf_counter()
    print("Converting adjacency matrix to neighbor lists...", flush=True)
    neighbors, degrees = adjacency_to_neighbors(adj)
    n_nodes = len(neighbors)
    print(f"Graph: nodes={n_nodes}, edge_rows={degrees.sum()}, isolated={(degrees == 0).sum()}; "
          f"conversion={time.perf_counter() - started:.2f}s", flush=True)

    node_neighbor = np.zeros((n_nodes, n_graphs, n_neighbors), dtype=np.int64)
    spatial_matrix = np.zeros((n_nodes, n_graphs, n_neighbors, n_neighbors), dtype=np.float32)
    print(f"Sampling {n_nodes * n_graphs} subgraphs; output arrays="
          f"{(node_neighbor.nbytes + spatial_matrix.nbytes) / 2**20:.1f} MiB", flush=True)

    for node_id in tqdm(range(n_nodes), desc="building sampled subgraphs"):
        for graph_id in range(n_graphs):
            walk = np.asarray(random_walk(node_id, neighbors, n_neighbors), dtype=np.int64)
            node_neighbor[node_id, graph_id] = walk
            spatial_matrix[node_id, graph_id] = shortest_path_in_graph(walk, neighbors)

    print(f"Graph preprocessing completed in {time.perf_counter() - started:.2f}s", flush=True)
    return node_neighbor, spatial_matrix, degrees
