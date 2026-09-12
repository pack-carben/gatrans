import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import ModelConfig
from models import TREE
from utils import binary_metrics, ensure_dirs, load_processed, seed_everything
from utils.data import NodeDataset, split_from_masks


def parse_args():
    parser = argparse.ArgumentParser(description="Train the GATrans/TREE classifier.")
    parser.add_argument("--dataset", default="BLCA")
    parser.add_argument("--n-graphs", type=int, default=6)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collect_logits(model, loader, device):
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for node_id, label in loader:
            logits = model(node_id.to(device))
            logits_all.append(logits.cpu().numpy())
            labels_all.append(label.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def main():
    args = parse_args()
    seed_everything(42)
    ensure_dirs()

    config = ModelConfig(
        dataset=args.dataset,
        n_graphs=args.n_graphs,
        n_neighbors=args.n_neighbors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    arrays = load_processed(config.processed_path)
    device = torch.device(args.device)

    train_id, y_train, val_id, y_val, test_id, y_test = split_from_masks(
        arrays["labels"],
        arrays["mask_train"],
        arrays["mask_val"],
        arrays["mask_test"],
    )
    positive_ratio = float(max(y_train.mean(), 1e-6))
    pos_weight = torch.tensor([(1.0 - positive_ratio) / positive_ratio], device=device)

    model = TREE(
        arrays["node_feature"],
        arrays["node_degree"],
        arrays["node_neighbor"],
        arrays["spatial_matrix"],
        n_graphs=config.n_graphs,
        n_neighbors=config.n_neighbors,
        d_model=config.d_model,
        n_layers=config.n_layers,
        num_heads=config.num_heads,
        dff=config.dff,
        d_sp_enc=config.d_sp_enc,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(NodeDataset(train_id, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(NodeDataset(val_id, y_val), batch_size=config.batch_size)
    test_loader = DataLoader(NodeDataset(test_id, y_test), batch_size=config.batch_size)

    best_auc = -1.0
    wait = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for node_id, label in train_loader:
            node_id = node_id.to(device)
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(node_id), label)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_metrics = binary_metrics(val_labels, val_logits)
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val={val_metrics}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            wait = 0
            torch.save({"model": model.state_dict(), "config": config.__dict__}, config.checkpoint_path)
        else:
            wait += 1
            if wait >= config.patience:
                break

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_metrics = binary_metrics(test_labels, test_logits)
    print(json.dumps({"best_val_auc": best_auc, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
