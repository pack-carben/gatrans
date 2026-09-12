import numpy as np


def build_cugraph_from_adjacency(adj: np.ndarray):
    """Build a cuGraph graph from a dense adjacency matrix.

    RAPIDS is intentionally imported inside the function so CPU environments can
    still import the project. Use this helper on the Linux CUDA server after
    installing the optional RAPIDS requirements.
    """

    try:
        import cudf
        import cugraph
    except ImportError as exc:
        raise RuntimeError(
            "RAPIDS/cuGraph is not installed. Install requirements-rapids-cu12.txt "
            "on a compatible Linux CUDA server before using --engine cugraph."
        ) from exc

    src, dst = np.where(adj > 0)
    edge_frame = cudf.DataFrame({"src": src.astype("int64"), "dst": dst.astype("int64")})
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(edge_frame, source="src", destination="dst", renumber=False)
    return graph, edge_frame
