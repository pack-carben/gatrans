import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig
from models import ShapFeatureWrapper, TREE
from utils.explain import build_shap_features, shap_feature_names
from utils.io import load_processed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and plot SHAP values for sampled subgraph features.")
    parser.add_argument("--dataset", default="BLCA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-graphs", type=int, default=6)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--background-size", type=int, default=64)
    parser.add_argument("--explain-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="logs/shap")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ModelConfig(dataset=args.dataset, n_graphs=args.n_graphs, n_neighbors=args.n_neighbors)
    arrays = load_processed(config.processed_path)
    device = torch.device(args.device)

    model = TREE(
        arrays["node_feature"],
        arrays["node_degree"],
        arrays["node_neighbor"],
        arrays["spatial_matrix"],
        n_graphs=args.n_graphs,
        n_neighbors=args.n_neighbors,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    train_ids = np.where(arrays["mask_train"])[0]
    explain_ids = train_ids[: args.explain_size]
    background_ids = train_ids[: args.background_size]

    background = build_shap_features(model, background_ids, device)
    explain_data = build_shap_features(model, explain_ids, device)
    wrapper = ShapFeatureWrapper(model, args.n_graphs, args.n_neighbors, arrays["node_feature"].shape[-1]).to(device)
    wrapper.eval()

    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_data)
    shap_values = np.asarray(shap_values).squeeze()
    importance = np.abs(shap_values).mean(axis=0)

    feature_names = arrays["feature_names"] if "feature_names" in arrays and arrays["feature_names"] is not None else [
        f"feature_{i}" for i in range(arrays["node_feature"].shape[-1])
    ]
    names = shap_feature_names(feature_names, args.n_graphs, args.n_neighbors)
    result = pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / f"{args.dataset}_shap_importance.csv", index=False)

    top = result.head(25).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.dataset}_shap_importance.png", dpi=200)
    print(f"Saved SHAP outputs to {out_dir}")


if __name__ == "__main__":
    main()
