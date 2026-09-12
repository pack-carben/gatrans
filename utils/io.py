import random
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch

from config import CHECKPOINT_DIR, LOG_DIR, PROCESSED_DIR


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in (PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_h5(path: Union[str, Path]) -> dict:
    with h5py.File(path, "r") as h5:
        data = {
            "network": h5["network"][:],
            "features": h5["features"][:].astype(np.float32),
            "y_train": h5["y_train"][:],
            "y_val": h5["y_val"][:] if "y_val" in h5 else None,
            "y_test": h5["y_test"][:],
            "mask_train": h5["mask_train"][:].astype(bool),
            "mask_val": h5["mask_val"][:].astype(bool) if "mask_val" in h5 else None,
            "mask_test": h5["mask_test"][:].astype(bool),
        }
        if "gene_names" in h5:
            data["gene_names"] = h5["gene_names"][:]
        if "feature_names" in h5:
            data["feature_names"] = h5["feature_names"][:]
    return data


def save_processed(path: Union[str, Path], **arrays) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_processed(path: Union[str, Path]) -> dict:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}
