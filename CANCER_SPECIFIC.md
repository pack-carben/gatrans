# Cancer-specific figure reproduction

## Current two-step workflow

No package installation or environment repair is part of either command.
Use the existing `/root/miniconda3` interpreter. Training does not import SHAP,
matplotlib, torchvision or ONNX and does not run environment preflight.

```bash
cd /root/autodl-tmp/gatrans
bash scripts/run_cancer_specific_all.sh
/root/miniconda3/bin/python scripts/shap_cancer_specific.py --run results/cancer_specific_all_v2
```

The first command trains all homogeneous and heterogeneous files. Each dataset
saves `graph.npz` (walks, distances, degrees), `inputs.npz` (features, labels,
gene/feature names), `run_config.json`, fold weights, splits and predictions.
The second command restores these saved arrays and fold-0 weights to calculate
SHAP without retraining. Older runs without `inputs.npz` require their original
training inputs; this new explanation entry point reports those runs as missing
inputs. The environment repair notes below describe earlier troubleshooting,
not required training steps.

For several independent datasets on one GPU, use the Python parallel launcher.
Its output layout is identical, but it defaults to a separate directory so it
cannot collide with a running serial job:

```bash
/root/miniconda3/bin/python scripts/train_cancer_specific_parallel.py --workers 2
/root/miniconda3/bin/python scripts/shap_cancer_specific.py --run results/cancer_specific_parallel
```

Each worker uses one CPU thread and one CUDA process. Start with two workers and
increase to four only if GPU memory remains comfortable. A lock file prevents
two parallel launchers from writing the same output directory.

This workflow evaluates **the PyTorch GATrans refactor** on the 16 homogeneous
and 15 heterogeneous cancer-specific datasets from TREE. It is not a claim that
the refactor is numerically identical to the TensorFlow architecture.

## Cloud storage and synchronization

Develop and commit locally, `git push`, then `git pull --ff-only` inside
`/root/autodl-tmp/gatrans`. Keep data, caches, weights, logs and results on
`/root/autodl-tmp`. The original compressed inputs remain in `/root/autodl-fs`.
Never put SSH passwords in code, configuration, logs or Git.

Extract HDF5 members only (excluding `__MACOSX`) into:

```
dataset/cancer_specific/homogeneous/*_multiomics.h5
dataset/cancer_specific/heterogeneous/*_multiomics.h5
```

Dependencies beyond the training environment: matplotlib, pandas, shap and requests.
The publisher XLSX reader uses the standard library and pandas; it supports Strict
OOXML directly, preserving the source files. Optional PDF inspection uses pypdf.

## Commands

The production entry point deliberately uses `/root/miniconda3/bin/python`,
requires all 31 source files, and writes results only to the data disk repository:

```bash
bash scripts/run_cancer_specific_all.sh --preflight-only
bash scripts/run_cancer_specific_all.sh
python scripts/plot_cancer_specific.py --run results/cancer_specific_all_v2
```

The preflight command reads and audits all 16 homogeneous and 15 heterogeneous
HDF5 files and runs a tiny synthetic forward/backward check. It does not build
real graph caches or train any dataset. If SciPy import raises a recursive NumPy
dtype traceback, repair the existing Miniconda environment once with
`bash scripts/repair_cloud_numeric_stack.sh`; this script rejects every prefix
except `/root/miniconda3` and never installs packages in `/root/autodl-tmp`.
The cloud Python 3.12 image was verified with NumPy 2.1.3 and SciPy 1.14.1;
the former NumPy 1.26.4 combination reproduced the FITPACK dtype recursion even
when importing scikit-learn without loading any project code.
The same repair also installs the official matching CUDA 12.1 family: PyTorch
2.4.1, torchvision 0.19.1 and torchaudio 2.4.1. Preflight imports `torch.onnx`
and torchvision compiled operators so mixed PyTorch releases are rejected before
any real dataset is trained.
The repair removes only residual `torch`, `torchvision` and `torchaudio`
directories under the validated `/root/miniconda3/.../site-packages` path after
pip uninstall. This is necessary because the cloud image contained unowned
PyTorch 2.8 exporter files that ordinary force-reinstall left in place.
When the numerical stack already passes and only PyTorch files are mixed, use
`bash scripts/repair_cloud_numeric_stack.sh --torch-only` to preserve NumPy,
SciPy, pandas and h5py.

Optional single-cancer integration check for debugging only (not a performance reproduction):

```bash
python -u scripts/cancer_specific.py --cancers BLCA --folds 1 --epochs 2 --shap-limit 4 --shap-samples 32 --out results/cancer_specific_smoke
python scripts/plot_cancer_specific.py --run results/cancer_specific_smoke
```

Full run: ten folds in the original training pool, at most 100 epochs per fold,
early stopping after 20 epochs without validation AP improvement. The original
test mask is never used for model selection. The HDF5 train and validation masks
are pooled before ten-fold CV; every generated split is saved. Fixed seed 42,
six channels, eight nodes per channel, three layers. No test-driven tuning.
These fixed hyperparameters do not reproduce the paper's grid search.

## Outputs and scope

- Fig. 2b: per-cancer published TREE and five baseline AUPRC values, new fold AP,
  fold SD, absolute difference, distributions across matching networks.
- Fig. 3a/b: original omics proportions/counts beside new fixed-graph SHAP from
  correctly classified held-out cancer genes. New confidence distributions and
  exact gene records are saved; original confidence samples are unavailable.
- Fig. 3e examples: original MUC1/KLF6/BAP1/CASP8 heatmaps beside cancer-specific
  model explanations. The source uses different pan-cancer feature splits.
  These panels are visual references, not direct error estimates.
- Additional plots: held-out ROC/PR, confidence scores, exploratory cancer-specific
  candidate heatmaps and attention matrices for available selected genes.
- Every plot is PNG and SVG with supporting CSV. SHAP features are also NPZ.
  Original workbooks, extracted tables and a URL/hash manifest stay in paper_sources.
  When a paper panel or a trained run lacks a cell needed to keep the visual
  layout aligned with the published figure, the plotting script writes separate
  `*_display.csv` tables with neutral display-only placeholders and
  `*_display_provenance.csv` cell audits. Proportion plots use 0.25 per omics
  channel for missing rows; signed SHAP and probability heatmaps use 0. The
  source/observed CSV files preserve missing values unchanged.

## Interpretation and reproducibility limits

Homogeneous HDF5 files have four actual columns but 64 feature names. We map them
to SNV/METH/GE/CNA for the target cancer and validate against identical values on
shared genes in the corresponding heterogeneous 64-column file. ESCA has no
heterogeneous counterpart; its mapping remains an explicit source-order assumption.
KIRC and THCA each have one gene with a gene-expression value differing between
the source archives. At least 99% of shared-gene values per column must match;
all exceptions and both values are recorded, and original inputs remain unchanged.

SHAP explains fold 0 by a predeclared rule; it does not select the best test fold.
Backgrounds contain only that fold's training nodes. Structure is held fixed and
the wrapper's prediction must match the trained classifier. Signed attributions
are summed over slots/features before taking magnitudes. The original publication
does not fully specify an identical cohort and aggregation. Additivity residuals
are stored and must be assessed before interpreting individual genes. Case studies
may include training genes and are excluded from held-out cohort statistics.

Heterogeneous inputs retain all 64 provided features and the supplied topology.
The current refactor uses a shared feature projection and untyped random walks,
not the original type-specific feature projections or dedicated metapath runs.
Distances are within sampled induced subgraphs. Attention is descriptive, not
causal; this workflow does not infer node types from gene names.

The published baseline numbers are not retrained baselines. AP and trapezoidal
AUPRC are both exported to expose metric-definition differences. Seeds, hashes,
Git revision, training histories, split IDs and all-node scores are preserved.
The workflow does not guarantee matching or exceeding the paper's accuracy.
Completion metadata distinguishes pilot runs from full fixed-configuration runs.
After a reviewed code fix, `--resume` can retain completed folds with unchanged
numerical arguments and data hashes. Original run provenance and new revision
events are both preserved; partially trained folds restart from their seed.
The full entry point records its exact 31-file manifest before work begins. A
dataset-level error is written to `status.json` and the remaining cancers still
run; the process exits nonzero after the complete collection has been attempted.
