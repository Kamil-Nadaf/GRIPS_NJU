"""CLI: run the GBM pipeline for one or more GRBs.

Example (inside Docker)::

    docker exec gbm python -m cli.run_grb GRB140606B
    docker exec gbm python -m cli.run_grb GRB140606B \\
        --stages geometry,spectra_tint,fit_tint,params --workers 4
    docker exec gbm python -m cli.run_grb GRB140606B \\
        --model band --include-bgo
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main(argv=None):
    from grb_config import ACTIVE_GRBS, grbs_df_all
    from pipeline.constants import DEFAULT_STAGES
    from pipeline.parallel import default_n_workers
    from pipeline.runner import GRBPipelineRunner, STAGE_ALIASES

    parser = argparse.ArgumentParser(
        prog='python -m cli.run_grb',
        description='Run the heapy+bayspec GBM pipeline (tint-first).')
    parser.add_argument(
        'grbs', nargs='*', default=None,
        help='GRB names (default: GRB_ACTIVE / GRB140606B)')
    parser.add_argument(
        '--stages', default=','.join(DEFAULT_STAGES),
        help=f'Comma-separated stages. Default: {",".join(DEFAULT_STAGES)}. '
             f'Aliases: {sorted(STAGE_ALIASES)}. Tres is opt-in '
             f'(spectra_tres, fit_tres).')
    parser.add_argument(
        '--model', default='cpl',
        help='bayspec additive model name (default: cpl). See --list-models.')
    parser.add_argument('--nlive', type=int, default=1000,
                        help='MultiNest live points (200 lecture / 1000 production)')
    parser.add_argument('--workers', type=int, default=None,
                        help='CPU workers (default GRB_N_WORKERS or min(cpu,4))')
    parser.add_argument('--force', action='store_true',
                        help='Rerun even if fingerprinted artifacts exist')
    parser.add_argument(
        '--include-bgo', action='store_true',
        help='Include BGO in spectral fits (default: NaI only)')
    parser.add_argument('--parallel-grbs', action='store_true',
                        help='Farm whole GRBs to separate processes')
    parser.add_argument('--list', action='store_true', help='Print catalog and exit')
    parser.add_argument('--list-models', action='store_true',
                        help='Print bayspec additive models and exit')
    args = parser.parse_args(argv)

    if args.list:
        print(grbs_df_all[['name', 'sel_dets', 't1', 't2', 'slice_mode']].to_string(index=False))
        return 0

    if args.list_models:
        from pipeline.fitting import available_models
        print('\n'.join(available_models()))
        return 0

    names = args.grbs or ACTIVE_GRBS
    n_workers = default_n_workers(args.workers)
    runner = GRBPipelineRunner(
        n_workers=n_workers, model_name=args.model, nlive=args.nlive,
        force=args.force, include_bgo=args.include_bgo)
    print(
        f'GRBs={names} stages={args.stages} workers={n_workers} '
        f'model={args.model} include_bgo={args.include_bgo}')
    if len(names) == 1:
        runner.run(names[0], stages=args.stages)
    else:
        runner.run_batch(names, stages=args.stages, parallel_grbs=args.parallel_grbs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
