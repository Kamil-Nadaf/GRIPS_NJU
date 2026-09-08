"""CLI: 3ML backend (writes under data/{tint,tres}/3ML/).

Example (inside Docker)::

    docker exec gbm python -m cli.run_3ml GRB140606B
    docker exec gbm python -m cli.run_3ml GRB140606B --mode tres --bin-method bayesblocks
    docker exec gbm python -m cli.run_3ml GRB140606B --model band --include-bgo
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main(argv=None):
    from grb_config import ACTIVE_GRBS
    from pipeline.context import GRBContext
    from pipeline.threeml import (
        BIN_METHODS, MODEL_ALIAS, run_3ml, threeml_available,
    )

    parser = argparse.ArgumentParser(
        prog='python -m cli.run_3ml',
        description='Run the 3ML GBM pipeline (sibling of heapy + bayspec).')
    parser.add_argument(
        'grbs', nargs='*', default=None,
        help='GRB names (default: GRB_ACTIVE / GRB140606B)')
    parser.add_argument(
        '--mode', choices=['tint', 'tintegrated', 'tres', 'tresolved'],
        default='tint', help='Time-integrated (default) or time-resolved')
    parser.add_argument(
        '--model', default='cpl',
        help=f'Spectral model. Aliases: {sorted(MODEL_ALIAS)}')
    parser.add_argument('--nlive', type=int, default=1000)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--include-bgo', action='store_true')
    parser.add_argument(
        '--bkg', default=None,
        help='Background intervals, e.g. "-20--0.83,5.74-55.74" '
             '(default: heapy LC pads minus burst)')
    parser.add_argument(
        '--bin-method', default='custom', choices=list(BIN_METHODS),
        help='Tres TimeSeriesBuilder binning (custom = ctx.time_slices). '
             'See Building_Plugins_from_TimeSeries.html')
    parser.add_argument('--dt', type=float, default=2.0, help='constant cadence dt (s)')
    parser.add_argument('--sigma', type=float, default=25.0, help='significance bin sigma')
    parser.add_argument('--p0', type=float, default=0.01, help='bayesblocks false-positive p0')
    parser.add_argument('--min-width', type=float, default=0.1, help='merge bins shorter than this (s)')
    parser.add_argument('--min-sigma', type=float, default=None, help='merge bins below this σ (if available)')
    args = parser.parse_args(argv)

    ok, info = threeml_available()
    if not ok:
        print(f'threeML unavailable: {info}', file=sys.stderr)
        return 1

    names = args.grbs or ACTIVE_GRBS
    mode = 'tresolved' if args.mode in ('tres', 'tresolved') else 'tintegrated'
    for name in names:
        ctx = GRBContext.from_name(name)
        print(f'3ML {name} mode={mode} model={args.model}')
        bin_kwargs = {
            'dt': args.dt, 'sigma': args.sigma, 'p0': args.p0,
            'min_width': args.min_width, 'min_sigma': args.min_sigma,
        }
        run_3ml(
            ctx, model_name=args.model, mode=mode, nlive=args.nlive,
            include_bgo=args.include_bgo, force=args.force,
            background_interval=args.bkg, bin_method=args.bin_method,
            bin_kwargs=bin_kwargs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
