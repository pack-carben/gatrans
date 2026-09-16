"""Read saved inputs, sampled graphs and fold-0 weights to explain all datasets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cancer_specific import build_model, explain, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/cancer_specific_all_full_sp')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--background', type=int, default=32)
    parser.add_argument('--shap-samples', type=int, default=200)
    parser.add_argument('--shap-limit', type=int, default=0)
    options = parser.parse_args()
    torch.set_num_threads(4)
    configs = sorted(Path(options.run).glob('*/*/run_config.json'))
    if not configs:
        parser.error('No saved training datasets found')
    failures = []
    for config in configs:
        dest = config.parent
        print(f'SHAP {dest.parent.name}/{dest.name}', flush=True)
        try:
            args = argparse.Namespace(**json.loads(config.read_text()))
            vars(args).update(vars(options))
            seed_everything(args.seed)
            with np.load(dest / 'inputs.npz') as saved:
                features, labels = saved['features'], saved['labels']
                genes, names = saved['genes'].tolist(), saved['feature_names'].tolist()
            with np.load(dest / 'graph.npz') as graph:
                arrays = dict(features=features, **{key: graph[key] for key in graph.files})
            with np.load(dest / 'fold_00/split.npz') as split:
                train_ids, test_ids = split['train'], split['test']
            model = build_model(arrays, args)
            weights = torch.load(dest / 'fold_00/weights.pt', weights_only=True, map_location=args.device)
            # Merge saved parameters with the graph buffers restored above.
            state = model.state_dict()
            state.update(weights)
            model.load_state_dict(state)
            explain(model, arrays, names, genes, labels, train_ids, test_ids, dest, args)
            del model
        except Exception as error:
            print(f'FAILED {dest}: {error}', file=sys.stderr, flush=True)
            failures.append(str(dest))
    if failures:
        raise RuntimeError('SHAP failed for: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
