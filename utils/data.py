import numpy as np
import torch
from torch.utils.data import Dataset


class NodeDataset(Dataset):
    """Dataset returning node ids and binary labels."""

    def __init__(self, node_ids: np.ndarray, labels: np.ndarray):
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.node_ids.numel())

    def __getitem__(self, index: int):
        return self.node_ids[index], self.labels[index]


def split_from_masks(labels: np.ndarray, mask_train, mask_val, mask_test):
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected one binary label per node.")
    masks = []
    for name, mask in (("train", mask_train), ("val", mask_val), ("test", mask_test)):
        if mask is None or np.asarray(mask).ndim == 0:
            raise ValueError(f"Missing {name} mask; provide an independent validation split.")
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(labels) or not mask.any():
            raise ValueError(f"Invalid or empty {name} mask.")
        if np.unique(labels[mask]).size != 2:
            raise ValueError(f"{name} split must contain both classes for AUC.")
        masks.append(mask)
    mask_train, mask_val, mask_test = masks
    if np.any(np.sum(masks, axis=0) > 1):
        raise ValueError("Train, validation and test masks overlap.")
    train_id = np.where(mask_train)[0]
    val_id = np.where(mask_val)[0] if mask_val is not None else train_id
    test_id = np.where(mask_test)[0]
    return (
        train_id,
        labels[train_id].astype(np.float32),
        val_id,
        labels[val_id].astype(np.float32),
        test_id,
        labels[test_id].astype(np.float32),
    )
