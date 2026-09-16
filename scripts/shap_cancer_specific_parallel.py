"""Calculate saved cancer-specific SHAP results in parallel GPU processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
import sys

# Keep independent SHAP workers from each creating a full CPU thread pool.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cancer_specific import write_json
from scripts.shap_cancer_specific import explain_dataset


def explain_one(config, options):
    torch.set_num_threads(1)
    return explain_dataset(config, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of datasets explained at the same time')
    parser.add_argument('--run', default='results/cancer_specific_parallel_full_sp')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--background', type=int, default=32)
    parser.add_argument('--shap-samples', type=int, default=200)
    parser.add_argument('--shap-limit', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if args.background < 1 or args.shap_samples < 1 or args.shap_limit < 0:
        parser.error('Invalid SHAP sizes')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA GPU is required for --device cuda')

    run = Path(args.run)
    configs = sorted(run.glob('*/*/run_config.json'))
    if not configs:
        parser.error(f'No saved training datasets found under {run}')
    lock = run / '.parallel_shap.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f'{lock} exists; another parallel SHAP run may be active')
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)

    options = {
        'run': args.run, 'device': args.device, 'background': args.background,
        'shap_samples': args.shap_samples, 'shap_limit': args.shap_limit,
    }
    completed, failures = [], []
    try:
        write_json(run / 'parallel_shap_config.json', {
            'workers': args.workers, 'datasets': len(configs), **options,
        })
        context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(explain_one, str(config), options): config for config in configs}
            for future in as_completed(futures):
                config = futures[future]
                dest = config.parent
                try:
                    result = future.result()
                    completed.append(result)
                    print(f"DONE {result['kind']}/{result['cancer']} "
                          f"({len(completed) + len(failures)}/{len(configs)})", flush=True)
                except Exception as error:
                    failure = {'kind': dest.parent.name, 'cancer': dest.name, 'error': str(error)}
                    failures.append(failure)
                    print(f"FAILED {failure['kind']}/{failure['cancer']}: {error}",
                          file=sys.stderr, flush=True)
                write_json(run / 'parallel_shap_status.json', {
                    'total': len(configs), 'completed': completed, 'failures': failures,
                })
    finally:
        lock.unlink(missing_ok=True)

    if failures:
        raise RuntimeError(f'{len(failures)} of {len(configs)} datasets failed; '
                           f'see {run / "parallel_shap_status.json"}')
    print(f'All {len(configs)} datasets explained.', flush=True)


if __name__ == '__main__':
    main()
