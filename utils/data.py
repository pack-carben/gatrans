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
