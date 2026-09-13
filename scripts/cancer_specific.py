"""Cancer-specific evaluation with untouched test masks and ten training folds.

This evaluates the PyTorch GATrans refactor, not the original TensorFlow TREE.
All outputs are kept under --out. Run on the cloud data disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import TREE
from utils.io import load_h5, seed_everything
from utils.data import split_from_masks
from scripts.show_shap import FixedGraphFeatures

OMICS = ['SNV', 'METH', 'GE', 'CNA']
GENES = ['MUC1', 'KLF6', 'BAP1', 'CASP8', 'BRCA1', 'SGK1', 'ERBB4', 'MYC', 'TP53', 'TET2']
BUFFERS = {'node_feature', 'node_degree', 'node_neighbor', 'spatial_matrix'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def decode(items):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in items]


def metrics(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    return {'auroc': float(roc_auc_score(labels, probs)),
            'average_precision': float(average_precision_score(labels, probs)),
            'auprc_trapezoid': float(auc(recall, precision))}


def feature_names(data, cancer, kind, root):
    names = decode(data['feature_names'])
    if len(names) == data['features'].shape[1]:
        return [n.replace('MF:', 'SNV:') for n in names], 'HDF5 feature names; MF renamed SNV'
    if kind != 'homogeneous' or data['features'].shape[1] != 4 or len(names) != 64:
        raise ValueError('Unrecognized feature-name mismatch')
    selected = [names.index(prefix + ': ' + cancer) for prefix in ['MF', 'METH', 'GE', 'CNA']]
    counterpart = root / 'heterogeneous' / (cancer + '_multiomics.h5')
    note = 'Four columns interpreted as cancer-specific SNV/METH/GE/CNA from source block order'
    if counterpart.exists():
        with h5py.File(counterpart) as h:
            lookup = {n: i for i, n in enumerate(decode(h['gene_names'][:]))}
            pairs = [(i, lookup[n]) for i, n in enumerate(decode(data['gene_names'])) if n in lookup]
            left = data['features'][[i for i, _ in pairs]]
            right = h['features'][:][[i for _, i in pairs]][:, selected]
            if not len(pairs) or not np.allclose(left, right, atol=1e-6):
                raise ValueError('Four-column mapping failed cross-network validation')
            note += f'; validated on {len(pairs)} matching genes against 64-column heterogeneous data'
    else:
        note += '; no heterogeneous ESCA counterpart exists; mapping is a documented assumption'
    return [n + ': ' + cancer for n in OMICS], note


def predict(model, ids, batch_size, device):
    model.eval()
    result = []
    with torch.no_grad():
        for chunk in np.array_split(ids, max(1, int(np.ceil(len(ids) / batch_size)))):
            result.extend(torch.sigmoid(model(torch.as_tensor(chunk, device=device))).cpu().tolist())
    return np.asarray(result)


def build_model(arrays, args):
    return TREE(arrays['features'], arrays['degree'], arrays['neighbors'], arrays['spatial'],
                n_graphs=args.channels, n_neighbors=args.neighbors, n_layers=args.layers,
                dropout=args.dropout).to(args.device)


def explain(model, arrays, names, genes, labels, train_ids, test_ids, out, args):
    import shap
    model.eval()
    rng = np.random.default_rng(args.seed)
    bg_ids = rng.choice(train_ids, min(args.background, len(train_ids)), replace=False)
    bg = model.node_feature[model.node_neighbor[torch.as_tensor(bg_ids, device=args.device)]]
    probs = predict(model, test_ids, args.batch_size, args.device)
    correct = test_ids[(labels[test_ids] == 1) & (probs >= 0.5)]
    if args.shap_limit > 0 and len(correct) > args.shap_limit:
        correct = rng.choice(correct, args.shap_limit, replace=False)
    target_ids = list(dict.fromkeys([int(i) for i in correct] +
                                   [i for i, g in enumerate(genes) if g in GENES]))
    rows, attributions = [], []
    groups = [OMICS.index(n.split(':')[0]) for n in names]
    for count, node in enumerate(target_ids):
        target = model.node_feature[model.node_neighbor[node]].unsqueeze(0)
        wrapper = FixedGraphFeatures(model, node).eval()
        with torch.no_grad():
            probability = wrapper(target).item()
            original = torch.sigmoid(model(torch.tensor([node], device=args.device))).item()
            if not np.isclose(probability, original, atol=1e-6):
                raise RuntimeError('Explanation wrapper differs from classifier')
            baseline = wrapper(bg).mean().item()
        explainer = shap.GradientExplainer(wrapper, bg, batch_size=min(32, args.batch_size))
        value = explainer.shap_values(target, nsamples=args.shap_samples, rseed=args.seed + count)
        if isinstance(value, list):
            value = value[0]
        value = np.asarray(value)
        if value.shape == (*target.shape, 1):
            value = value[..., 0]
        if value.shape != tuple(target.shape) or not np.isfinite(value).all():
            raise RuntimeError('Invalid SHAP values')
        signed = value[0].sum(axis=(0, 1))
        # Aggregate additive signed contributions first, then magnitude per omics.
        omics = np.array([signed[np.array(groups) == i].sum() for i in range(4)])
        row = dict(node_id=node, gene=genes[node], label=int(labels[node]), prediction=probability,
                   cohort='test_true_positive' if node in correct else 'case_study',
                   split='test' if node in test_ids else ('train' if node in train_ids else 'validation_or_unlabelled'),
                   baseline=baseline, residual=probability-baseline-float(value.sum()))
        row.update({f'shap_{o}': float(v) for o, v in zip(OMICS, omics)})
        rows.append(row)
        attributions.append(signed)
        print(f'SHAP {count+1}/{len(target_ids)} {genes[node]} residual={row["residual"]:.4f}', flush=True)
    pd.DataFrame(rows, columns=['node_id', 'gene', 'label', 'prediction', 'cohort', 'split', 'baseline', 'residual'] +
                 ['shap_' + o for o in OMICS]).to_csv(out / 'shap_genes.csv', index=False)
    np.savez_compressed(out / 'shap_features.npz', values=np.asarray(attributions).reshape(-1, len(names)),
                        node_ids=target_ids, genes=np.asarray([genes[i] for i in target_ids]),
                        feature_names=np.asarray(names), background_ids=bg_ids)
    write_json(out / 'explanation.json', dict(fold=0, method='SHAP GradientExplainer, fixed target graph',
               output='probability', aggregation='signed sums across slots and feature columns per omics',
               background='fold 0 training nodes only', background_size=len(bg_ids), nsamples=args.shap_samples,
               correct_test_positives_total=int(((labels[test_ids] == 1) & (probs >= .5)).sum()),
               correct_test_positives_explained=len(correct), case_studies='may include training genes; excluded from cohort statistics'))
    # Attention is a descriptive statistic, not a causal effect. Save graph 0, all layers.
    att_rows = []
    for gene in ('TP53', 'TET2', 'BRCA1', 'MYC'):
        if gene not in genes:
            continue
        node = genes.index(gene)
        with torch.no_grad():
            ids = model.node_neighbor[node:node+1]
            _, attention = model._encode_channels(model.node_feature[ids], model.node_degree[ids],
                                                 model.spatial_matrix[node:node+1])
        for layer in range(args.layers):
            weights = attention[layer][0].mean(dim=0).cpu().numpy()
            slots = ids[0, 0].cpu().numpy()
            distances = model.spatial_matrix[node, 0].cpu().numpy()
            for a in range(args.neighbors):
                for b in range(args.neighbors):
                    att_rows.append(dict(gene=gene, layer=layer+1, channel=0, source_slot=a, target_slot=b,
                                         source=genes[slots[a]], target=genes[slots[b]],
                                         distance=float(distances[a, b]), attention=float(weights[a, b])))
    pd.DataFrame(att_rows, columns=['gene', 'layer', 'channel', 'source_slot', 'target_slot', 'source', 'target', 'distance', 'attention']).to_csv(out / 'attention.csv', index=False)


def run_dataset(path, out, args):
    kind, cancer = path.parent.name, path.name.split('_')[0]
    dest = out / kind / cancer
    dest.mkdir(parents=True, exist_ok=True)
    signature = {k: v for k, v in vars(args).items() if k not in ('out', 'cancers', 'kinds')}
    with path.open('rb') as source:
        signature['data_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    signature['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    signature_file = dest / 'run_config.json'
    if signature_file.exists() and json.loads(signature_file.read_text()) != signature:
        raise ValueError(f'Run configuration changed; use a different output directory: {dest}')
    write_json(signature_file, signature)
    if (dest / 'complete.json').exists():
        print(f'Already complete: {kind}/{cancer}', flush=True)
        return
    started = time.time()
    data = load_h5(path)
    names, naming_note = feature_names(data, cancer, kind, Path(args.data))
    labels = data['y_train'].reshape(-1).copy()
    for split in ('val', 'test'):
        labels[data['mask_' + split]] = data['y_' + split].reshape(-1)[data['mask_' + split]]
    tr, _, va, _, test_ids, _ = split_from_masks(labels, data['mask_train'], data['mask_val'], data['mask_test'])
    pool = np.sort(np.concatenate([tr, va]))
    genes = decode(data['gene_names'])
    if len(set(genes)) != len(genes):
        raise ValueError('Duplicate gene names: grouped splitting required')
    if not np.isfinite(data['features']).all():
        raise ValueError('Non-finite input features')
    splits = list(StratifiedKFold(10, shuffle=True, random_state=args.seed).split(pool, labels[pool]))
    if min(np.bincount(labels[pool].astype(int))) < 10:
        raise ValueError('Insufficient class members for ten-fold CV')
    write_json(dest / 'data_audit.json', dict(network_kind=kind, cancer=cancer, nodes=len(genes),
               features=len(names), naming_note=naming_note, feature_names=names,
               train_pool=len(pool), test=len(test_ids), test_positive=int(labels[test_ids].sum()),
               input_file=str(path.resolve()), graph_sampling='untyped random walks; induced-subgraph shortest paths',
               model='PyTorch GATrans refactor; shared projection; not original TensorFlow TREE'))
    cache = dest / 'graph.npz'
    if cache.exists():
        with np.load(cache) as z:
            neighbors, spatial, degree = z['neighbors'], z['spatial'], z['degree']
    else:
        from utils.cuda_preprocess import build_subgraphs_cuda
        from utils.preprocess import build_subgraphs
        builder = build_subgraphs_cuda if args.device.startswith('cuda') else build_subgraphs
        neighbors, spatial, degree = builder(data['network'], args.channels, args.neighbors, args.seed)
        np.savez_compressed(cache, neighbors=neighbors, spatial=spatial, degree=degree)
    arrays = dict(features=data['features'], neighbors=neighbors, spatial=spatial, degree=degree)
    del data['network']
    fold_predictions = []
    for fold, (train_index, val_index) in enumerate(splits[:args.folds]):
        fold_dir = dest / f'fold_{fold:02d}'
        fold_dir.mkdir(exist_ok=True)
        train_ids, val_ids = pool[train_index], pool[val_index]
        np.savez_compressed(fold_dir / 'split.npz', train=train_ids, val=val_ids, test=test_ids)
        if (fold_dir / 'metrics.json').exists():
            fold_predictions.append(np.load(fold_dir / 'predictions.npy'))
            continue
        seed_everything(args.seed + fold)
        model = build_model(arrays, args)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-7)
        ratio = labels[train_ids].mean()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-ratio)/ratio, device=args.device))
        best, wait, history = -1., 0, []
        best_path = fold_dir / 'weights.pt'
        for epoch in range(args.epochs):
            model.train()
            order = np.random.permutation(train_ids)
            losses = []
            for start in range(0, len(order), args.batch_size):
                ids = order[start:start+args.batch_size]
                optim.zero_grad(set_to_none=True)
                loss = criterion(model(torch.as_tensor(ids, device=args.device)),
                                 torch.as_tensor(labels[ids], device=args.device, dtype=torch.float32))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss')
                loss.backward()
                optim.step()
                losses.append(float(loss.detach()))
            val = metrics(labels[val_ids], predict(model, val_ids, args.batch_size, args.device))
            history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), **val))
            if val['average_precision'] > best:
                best, wait = val['average_precision'], 0
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if k not in BUFFERS}, best_path)
            else:
                wait += 1
            print(f'{kind}/{cancer} fold={fold+1}/{args.folds} epoch={epoch+1} val_AP={val["average_precision"]:.4f}', flush=True)
            if wait >= args.patience:
                break
        model.load_state_dict(torch.load(best_path, weights_only=True, map_location=args.device), strict=False)
        probability = predict(model, np.arange(len(genes)), args.batch_size, args.device)
        np.save(fold_dir / 'predictions.npy', probability)
        pd.DataFrame(history).to_csv(fold_dir / 'history.csv', index=False)
        write_json(fold_dir / 'metrics.json', dict(fold=fold, best_val_ap=best, epochs=len(history),
                   **metrics(labels[test_ids], probability[test_ids])))
        fold_predictions.append(probability)
        del model, optim
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    preds = np.asarray(fold_predictions)
    pd.DataFrame(dict(node_id=np.arange(len(genes)), gene=genes, label=labels,
                     split=np.where(np.isin(np.arange(len(genes)), test_ids), 'test',
                           np.where(np.isin(np.arange(len(genes)), pool), 'training_pool', 'unlabelled')),
                     prediction=preds.mean(axis=0), prediction_sd=preds.std(axis=0))).to_csv(dest / 'predictions.csv', index=False)
    if args.shap_samples > 0:
        model = build_model(arrays, args)
        model.load_state_dict(torch.load(dest / 'fold_00/weights.pt', weights_only=True, map_location=args.device), strict=False)
        explain(model, arrays, names, genes, labels, pool[splits[0][0]], test_ids, dest, args)
        del model
    write_json(dest / 'complete.json', dict(folds=args.folds, seconds=time.time()-started,
               status='full_fixed_configuration' if args.folds == 10 and args.epochs == 100 else 'pilot',
               hyperparameter_search=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='dataset/cancer_specific')
    p.add_argument('--out', default='results/cancer_specific_v1')
    p.add_argument('--kinds', nargs='+', default=['homogeneous', 'heterogeneous'])
    p.add_argument('--cancers', nargs='+')
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--channels', type=int, default=6)
    p.add_argument('--neighbors', type=int, default=8)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=.5)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--background', type=int, default=32)
    p.add_argument('--shap-samples', type=int, default=200)
    p.add_argument('--shap-limit', type=int, default=0, help='0 explains every correctly predicted test cancer gene')
    args = p.parse_args()
    if not 1 <= args.folds <= 10 or min(args.epochs, args.patience, args.channels, args.neighbors, args.layers, args.batch_size, args.background) < 1:
        p.error('Invalid run sizes')
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [path for kind in args.kinds for path in sorted((Path(args.data) / kind).glob('*_multiomics.h5'))
             if not args.cancers or path.name.split('_')[0] in args.cancers]
    if not files:
        raise FileNotFoundError('No matching datasets')
    for path in files:
        write_json(out / 'status.json', dict(status='running', dataset=str(path), total_datasets=len(files)))
        run_dataset(path, out, args)
    write_json(out / 'status.json', dict(status='completed', datasets=len(files)))


if __name__ == '__main__':
    main()
