from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
PROCESSED_DIR = ROOT_DIR / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"


@dataclass
class ModelConfig:
    """Configuration shared by preprocessing, training, and explanation."""

    dataset: str = "BLCA"
    n_graphs: int = 6
    n_neighbors: int = 8
    d_model: int = 64
    n_layers: int = 3
    num_heads: int = 4
    dff: int = 128
    d_sp_enc: int = 128
    dropout: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 5e-7
    batch_size: int = 64
    epochs: int = 100
    patience: int = 20
    positive_ratio: Optional[float] = None

    @property
    def processed_path(self) -> Path:
        return PROCESSED_DIR / f"{self.dataset}_graphs_{self.n_graphs}_neighbors_{self.n_neighbors}.npz"

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.dataset}_best.pt"
