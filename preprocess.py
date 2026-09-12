import argparse
import time

from config import DATASET_DIR, ModelConfig
from utils.io import ensure_dirs, load_h5, save_processed, seed_everything
from utils.preprocess import build_subgraphs


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare sampled TREE/GATrans subgraphs.")
    parser.add_argument("--dataset", default="BLCA")
    parser.add_argument("--h5", default=None)
    parser.add_argument("--n-graphs", type=int, default=6)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine", choices=["cpu", "cuda", "cugraph"], default="cpu")
    parser.add_argument("--graph-batch-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dirs()

    config = ModelConfig(dataset=args.dataset, n_graphs=args.n_graphs, n_neighbors=args.n_neighbors)
    h5_path = args.h5 or DATASET_DIR / f"{args.dataset}_multiomics.h5"
    started = time.perf_counter()
    print(f"Loading HDF5: {h5_path}...", flush=True)
    data = load_h5(h5_path)
    print(f"Loaded in {time.perf_counter() - started:.2f}s; "
          f"network={data['network'].shape}, features={data['features'].shape}", flush=True)
    if args.engine == "cugraph":
        from utils.cugraph_backend import build_cugraph_from_adjacency

        _, edge_frame = build_cugraph_from_adjacency(data["network"])
        print(f"cuGraph graph created with {len(edge_frame)} directed edge rows.")
        print("Using the compatible sampled-subgraph writer; replace build_subgraphs when RAPIDS version is fixed.")

    builder = build_subgraphs
    builder_options = {}
    if args.engine == "cuda":
        from utils.cuda_preprocess import build_subgraphs_cuda

        builder = build_subgraphs_cuda
        builder_options["batch_size"] = args.graph_batch_size

    node_neighbor, spatial_matrix, node_degree = builder(
        data["network"],
        n_graphs=args.n_graphs,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
        **builder_options,
    )

    y = data["y_train"].copy()
    if data["y_val"] is not None:
        y[data["mask_val"]] = data["y_val"][data["mask_val"]]
    y[data["mask_test"]] = data["y_test"][data["mask_test"]]

    arrays = {
        "node_feature": data["features"],
        "node_degree": node_degree,
        "node_neighbor": node_neighbor,
        "spatial_matrix": spatial_matrix,
        "labels": y.astype("float32"),
        "mask_train": data["mask_train"],
        "mask_val": data["mask_val"],
        "mask_test": data["mask_test"],
    }
    if data.get("feature_names") is not None:
        arrays["feature_names"] = data["feature_names"]
    if data.get("gene_names") is not None:
        arrays["gene_names"] = data["gene_names"]

    print(f"Compressing and saving {config.processed_path}...", flush=True)
    save_processed(config.processed_path, **arrays)
    print(f"Saved processed data to {config.processed_path}")


if __name__ == "__main__":
    main()
