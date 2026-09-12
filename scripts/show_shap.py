"""Feature-only expected-gradient SHAP, with each target's graph fixed."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ModelConfig
from models import TREE
from utils.io import seed_everything


class FixedGraphFeatures(torch.nn.Module):
    def __init__(self, model, node):
        super().__init__()
        self.model = model
        self.register_buffer("degree", model.node_degree[model.node_neighbor[node]].unsqueeze(0))
        self.register_buffer("spatial", model.spatial_matrix[node].unsqueeze(0))

    def forward(self, features):
        embedding, _ = self.model.forward_features(
            features, self.degree.expand(features.shape[0], -1, -1),
            self.spatial.expand(features.shape[0], -1, -1, -1))
        return torch.sigmoid(self.model.classifier(embedding))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n-graphs", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--explain-size", type=int, default=16)
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-names", nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="logs/shap")
    args = parser.parse_args()
    if min(args.background_size, args.explain_size, args.nsamples, args.batch_size) < 1:
        raise ValueError("Sample counts and batch size must be positive.")
    started = time.perf_counter()
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    for name in ("dataset", "n_graphs", "n_neighbors"):
        if getattr(args, name) is not None and getattr(args, name) != getattr(config, name):
            raise ValueError(f"{name} conflicts with checkpoint configuration.")
    with np.load(config.processed_path, allow_pickle=True) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    state = checkpoint["model"]
    model = TREE(state["node_feature"], state["node_degree"], state["node_neighbor"],
                 state["spatial_matrix"], **{k: getattr(config, k) for k in
                 ("n_graphs", "n_neighbors", "d_model", "n_layers", "num_heads", "dff", "d_sp_enc", "dropout")}).to(device)
    model.load_state_dict(state)
    model.eval()
    n_nodes, feature_dim = model.node_feature.shape
    if arrays["labels"].shape[0] != n_nodes:
        raise ValueError("Metadata and checkpoint have different node counts.")
    raw_names = arrays.get("feature_names", [])
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in raw_names]
    naming_note = "Names from processed data."
    if args.feature_names:
        names = args.feature_names
        naming_note = "Names supplied explicitly."
        if len(names) != feature_dim:
            raise ValueError("Supply exactly one name per feature column.")
    elif len(names) != feature_dim:
        naming_note = f"Metadata has {len(names)} names for {feature_dim} columns; using generic names."
        names = [f"feature_{i}" for i in range(feature_dim)]
        print(naming_note, flush=True)
    rng = np.random.default_rng(args.seed)
    train_ids = np.flatnonzero(arrays["mask_train"])
    test_ids = np.flatnonzero(arrays["mask_test"])
    if not len(train_ids) or not len(test_ids) or np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Need nonempty disjoint training and test masks.")
    background_ids = rng.choice(train_ids, min(args.background_size, len(train_ids)), replace=False)
    explain_ids = rng.choice(test_ids, min(args.explain_size, len(test_ids)), replace=False)
    background = model.node_feature[model.node_neighbor[torch.as_tensor(background_ids, device=device)]]
    values, predictions, baselines, residuals = [], [], [], []
    print(f"Checkpoint={checkpoint_path}; configured epochs={config.epochs}; device={device}", flush=True)
    print(f"SHAP: background={len(background_ids)}, test nodes={len(explain_ids)}, nsamples={args.nsamples}; fixed graph", flush=True)
    for index, node_id in enumerate(tqdm(explain_ids, desc="SHAP " + str(device), unit="node")):
        target = model.node_feature[model.node_neighbor[int(node_id)]].unsqueeze(0)
        wrapper = FixedGraphFeatures(model, int(node_id)).eval()
        with torch.no_grad():
            prediction = wrapper(target).item()
            original = torch.sigmoid(model(torch.tensor([int(node_id)], device=device))).item()
            if not np.isclose(prediction, original, atol=1e-6):
                raise RuntimeError("SHAP wrapper does not reproduce model prediction.")
            baseline = wrapper(background).mean().item()
        explainer = shap.GradientExplainer(wrapper, background, batch_size=args.batch_size)
        attribution = explainer.shap_values(target, nsamples=args.nsamples, rseed=args.seed + index)
        if isinstance(attribution, list):
            attribution = attribution[0]
        attribution = np.asarray(attribution)
        if attribution.shape == (*target.shape, 1):
            attribution = attribution[..., 0]
        if attribution.shape != tuple(target.shape) or not np.isfinite(attribution).all():
            raise RuntimeError(f"Invalid SHAP output: {attribution.shape}")
        values.append(attribution[0])
        predictions.append(prediction)
        baselines.append(baseline)
        residuals.append(prediction - baseline - float(attribution.sum()))
    values = np.asarray(values)
    grouped = values.sum(axis=(1, 2))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = config.dataset
    importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(grouped).mean(axis=0),
                               "mean_signed_shap": grouped.mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out / f"{prefix}_shap_importance.csv", index=False)
    slot_names = [f"graph{g}_node{n}_{name}" for g in range(config.n_graphs)
                  for n in range(config.n_neighbors) for name in names]
    pd.DataFrame({"feature": slot_names, "mean_abs_shap": np.abs(values).mean(axis=0).reshape(-1)}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(out / f"{prefix}_shap_slots.csv", index=False)
    records = pd.DataFrame({"node_id": explain_ids, "label": arrays["labels"].reshape(-1)[explain_ids],
                            "prediction": predictions, "background_prediction": baselines,
                            "attribution_sum": values.sum(axis=(1, 2, 3)), "additivity_residual": residuals})
    if "gene_names" in arrays:
        records["gene"] = [x.decode() if isinstance(x, bytes) else str(x) for x in arrays["gene_names"][explain_ids]]
    for column, name in enumerate(names):
        records[f"shap_{name}"] = grouped[:, column]
    records.to_csv(out / f"{prefix}_shap_samples.csv", index=False)
    np.savez_compressed(out / f"{prefix}_shap_values.npz", shap_values=values, node_ids=explain_ids,
                        background_ids=background_ids, feature_names=np.asarray(names), predictions=predictions,
                        baseline_predictions=baselines, additivity_residuals=residuals)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_data = importance.iloc[::-1]
    ax.barh(plot_data["feature"], plot_data["mean_abs_shap"], color="#287D8E")
    ax.set_xlabel("Mean |summed feature attribution| (probability)")
    ax.set_title(f"{prefix}: feature SHAP, graph fixed ({len(explain_ids)} test nodes)")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_shap_importance.png", dpi=180)
    plt.close(fig)
    report = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
              "configured_epochs": config.epochs, "device": str(device), "background_size": len(background_ids),
              "explain_size": len(explain_ids), "nsamples": args.nsamples, "seed": args.seed,
              "method": "Expected gradients; graph fixed per target; probability output",
              "aggregation": "Sum signed slot contributions across channels/neighbors, then mean absolute value across targets",
              "feature_names_note": naming_note, "mean_abs_additivity_residual": float(np.abs(residuals).mean()),
              "max_abs_additivity_residual": float(np.abs(residuals).max()),
              "elapsed_seconds": time.perf_counter() - started}
    (out / f"{prefix}_shap_run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(importance.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved SHAP outputs to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
