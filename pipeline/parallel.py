"""ProcessPoolExecutor helpers (CPU; heapy/bayspec are not GPU-bound)."""

import os
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed


def default_n_workers(n_workers=None):
    if n_workers is not None:
        return max(1, int(n_workers))
    env = os.environ.get('GRB_N_WORKERS')
    if env:
        return max(1, int(env))
    return min(os.cpu_count() or 1, 4)


def map_parallel(fn, jobs, n_workers=None, desc=None):
    """Run ``fn(job)`` over ``jobs``. Sequential if ``n_workers==1`` or one job.

    If a worker is killed (heapy/OpenMP segfault → ``BrokenProcessPool``),
    unfinished jobs are retried in the parent process so one detector does
    not abort the whole extraction.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    workers = default_n_workers(n_workers)
    if workers == 1 or len(jobs) == 1:
        return [_run_one(fn, job, desc, i, len(jobs)) for i, job in enumerate(jobs)]

    results = [None] * len(jobs)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(fn, job): i for i, job in enumerate(jobs)}
            for fut in as_completed(future_map):
                i = future_map[fut]
                try:
                    results[i] = fut.result()
                    if desc:
                        print(f'  [{desc} {i + 1}/{len(jobs)} done]')
                except BrokenExecutor as exc:
                    print(f'  WARNING: worker crashed on job {i} ({exc})')
                    results[i] = None
    except BrokenExecutor as exc:
        print(f'  WARNING: process pool broke ({exc}); '
              'retrying unfinished jobs sequentially')

    for i, job in enumerate(jobs):
        if results[i] is not None:
            continue
        print(f'  [{desc or "job"} {i + 1}/{len(jobs)} sequential retry]')
        try:
            results[i] = _run_one(fn, job, None, i, len(jobs))
        except Exception as exc:
            print(f'  ERROR sequential job {i}: {exc}')
            payload = job if isinstance(job, dict) else {}
            results[i] = {
                'ok': False, 'error': str(exc),
                'det': payload.get('det'),
            }
    return results


def _run_one(fn, job, desc, i, n):
    if desc:
        print(f'  [{desc} {i + 1}/{n}]')
    return fn(job)
