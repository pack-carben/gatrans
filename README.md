# GATrans

GATrans is a PyTorch refactor of the TREE-style pan-cancer gene identification
workflow. It keeps the original project concepts while using a cleaner module
layout and GPU-friendly tensor execution.

## Layout

```text
config.py              Runtime and model configuration
train.py               Training and validation entrypoint
preprocess.py          H5-to-subgraph preprocessing entrypoint
models/tree.py         TREE classifier
layers/                Graphormer and attention layers
utils/                 I/O, preprocessing, metrics, SHAP helpers
scripts/show_shap.py   Example SHAP feature attribution script
```

## Basic Usage

```bash
python preprocess.py --dataset BLCA --h5 dataset/BLCA_multiomics.h5
python train.py --dataset BLCA
python scripts/show_shap.py --dataset BLCA --checkpoint checkpoints/BLCA_best.pt
```

RAPIDS/cuGraph support is optional and should be installed only on a compatible
Linux CUDA server with `requirements-rapids-cu12.txt`.
