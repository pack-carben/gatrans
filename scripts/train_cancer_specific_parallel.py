"""Train all cancer-specific datasets with several independent GPU workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

# Prevent each worker from creating a full CPU thread pool.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cancer_specific import discover_datasets, run_dataset, write_json


def train_one(path, output, config):
    """Run one dataset inside a spawned process."""
    torch.set_num_threads(1)
    args = argparse.Namespace(**config)
    path = Path(path)
    result = run_dataset(path, Path(output), args)
    return {'kind': path.parent.name, 'cancer': path.name.split('_')[0], 'status': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of datasets trained at the same time (start with 2; try up to 4)')
    parser.add_argument('--data', default='dataset/cancer_specific')
    parser.add_argument('--out', default='results/cancer_specific_parallel_full_sp')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if not torch.cuda.is_available():
        parser.error('CUDA GPU is required')

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.parallel.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f'{lock} exists; another parallel run may be active')
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)

    try:
        config = dict(
            data=args.data, out=args.out, kinds=['homogeneous', 'heterogeneous'], cancers=None,
            folds=10, epochs=100, patience=20, channels=6, neighbors=8, layers=3,
            dropout=.5, lr=.001, batch_size=64, seed=42, device='cuda',
            background=32, shap_samples=0, shap_limit=0, resume=False,
            require_full_dataset=True, preflight_only=False, fail_fast=False,
        )
        files = discover_datasets(args.data, config['kinds'], require_full=True)
        write_json(output / 'parallel_config.json', {
            'workers': args.workers, 'datasets': len(files), 'training': config,
        })

        completed, failures = [], []
        context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(train_one, str(path), str(output), config): path for path in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    completed.append(result)
                    print(f"DONE {result['kind']}/{result['cancer']} "
                          f"({len(completed) + len(failures)}/{len(files)})", flush=True)
                except Exception as error:
                    failure = {'kind': path.parent.name, 'cancer': path.name.split('_')[0],
                               'error': str(error)}
                    failures.append(failure)
                    print(f"FAILED {failure['kind']}/{failure['cancer']}: {error}",
                          file=sys.stderr, flush=True)
                write_json(output / 'parallel_status.json', {
                    'total': len(files), 'completed': completed, 'failures': failures,
                })
    finally:
        lock.unlink(missing_ok=True)

    if failures:
        raise RuntimeError(f'{len(failures)} of {len(files)} datasets failed; '
                           f'see {output / "parallel_status.json"}')
    print(f'All {len(files)} datasets completed.', flush=True)


if __name__ == '__main__':
    main()
