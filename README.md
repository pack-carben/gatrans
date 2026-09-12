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

Preprocessing reports HDF5 loading, graph size, isolated nodes, output memory,
sampling progress, and compression. Training shows batch progress, loss,
validation metrics, epoch duration, and checkpoint saves. Use unbuffered Python
when capturing output on a server:

```bash
mkdir -p logs
python -u preprocess.py --dataset BLCA > logs/preprocess.log 2>&1
python -u preprocess.py --dataset BLCA --engine cuda --graph-batch-size 512 > logs/preprocess_cuda.log 2>&1
python -u train.py --dataset BLCA --device cuda --epochs 3 > logs/train_smoke.log 2>&1
tail -f logs/train_smoke.log
python -m unittest discover -s tests -v
```

Labels may have shape `(N,)` or `(N, 1)`. Train, validation, and test masks must
be nonempty, disjoint, and each contain both classes for ROC AUC evaluation.
Missing validation masks are rejected instead of evaluating on the training set.
Regenerate processed data after updating the sampled shortest-path calculation:
repeated occurrences of the same node now have distance zero and identical
distances to other nodes.

RAPIDS/cuGraph support is optional and should be installed only on a compatible
Linux CUDA server with `requirements-rapids-cu12.txt`.
The current `--engine cugraph` option only constructs a GPU graph; sampled walks
and shortest-path calculations still use the CPU implementation.
Use `--engine cuda` for actual GPU random-walk sampling and batched shortest-path
computation through PyTorch, without RAPIDS. Reduce `--graph-batch-size` if batch
memory is limited; the full adjacency and its neighbor indices must still fit
in GPU memory. CPU and CUDA use different random generators, so sampled walks
need not match across engines; distances for identical walks are tested to agree.
