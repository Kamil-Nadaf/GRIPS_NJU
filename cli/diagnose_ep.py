"""Diagnose GRB140606B CPL Ep vs Yan et al. 2024 / Fermi catalog.

Run inside Docker::

    python /workspace/cli/diagnose_ep.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.mpl_setup import silence_missing_fonts
silence_missing_fonts()

OUTDIR = '/workspace/data/GRB140606B/diagnostics'


def _pha_rate(path):
    with fits.open(path) as hdul:
        names = [h.name for h in hdul]
        data = hdul['SPECTRUM'].data if 'SPECTRUM' in names else hdul[1].data
        cols = data.columns.names
        counts = np.asarray(data['COUNTS'] if 'COUNTS' in cols else data['RATE'], dtype=float)
        if counts.ndim > 1:
            counts = counts[0]
        exposure = float(hdul[1].header.get('EXPOSURE', hdul[0].header.get('EXPOSURE', 1.0)))
        return counts, exposure, cols, names


def _ebounds(rsp_path):
    with fits.open(rsp_path) as hdul:
        eb = hdul['EBOUNDS'].data
        return np.asarray(eb['E_MIN'], float), np.asarray(eb['E_MAX'], float)


def load_spec(base):
    src, tsrc, _, _ = _pha_rate(base + '.src')
    bkg, tbkg, _, _ = _pha_rate(base + '.bkg')
    emin, emax = _ebounds(base + '.rsp')
    e = 0.5 * (emin + emax)
    de = np.maximum(emax - emin, 1e-6)
    # rates per keV
    src_rate = src / tsrc
    bkg_scale = (src / tsrc)  # placeholder
    bkg_rate = bkg / tbkg
    net = src_rate - bkg_rate
    # Poisson-ish S/N on counts scaled to src exposure
    bkg_in_src = bkg_rate * tsrc
    snr = net * tsrc / np.sqrt(np.maximum(src + bkg_in_src, 1.0))
    return {
        'e': e, 'de': de, 'emin': emin, 'emax': emax,
        'src': src, 'bkg': bkg, 'tsrc': tsrc, 'tbkg': tbkg,
        'src_rate': src_rate, 'bkg_rate': bkg_rate, 'net': net, 'snr': snr,
    }


def band_snr(spec, lo, hi):
    m = (spec['e'] >= lo) & (spec['e'] <= hi)
    if not np.any(m):
        return 0.0, 0.0, 0.0
    tsrc, tbkg = spec['tsrc'], spec['tbkg']
    src = spec['src'][m].sum()
    bkg = spec['bkg'][m].sum() * (tsrc / tbkg)
    net = src - bkg
    snr = net / np.sqrt(max(src + bkg, 1.0))
    return float(net), float(snr), float(src)


def plot_count_spec(specs, title, out):
    fig, axes = plt.subplots(len(specs), 1, figsize=(8, 2.4 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]
    for ax, (label, spec) in zip(axes, specs):
        e = spec['e']
        ax.step(e, spec['src_rate'] / spec['de'], where='mid', label='src', color='#1f77b4')
        ax.step(e, spec['bkg_rate'] / spec['de'], where='mid', label='heapy bkg', color='#d62728')
        ax.step(e, np.clip(spec['net'], 0, None) / spec['de'], where='mid',
                label='net', color='#2ca02c')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('counts s$^{-1}$ keV$^{-1}$')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(5, 4e4)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Energy (keV)')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def offpulse_vs_heapy(ctx, det='n3'):
    """Compare heapy .bkg to a simple off-pulse mean (paper-like channel baseline)."""
    from pipeline.lightcurve import prepare_tte_for_lc
    from pipeline.spectra import tint_spec_base

    gbm_tte, offset = prepare_tte_for_lc(ctx, det)
    ev = np.asarray(gbm_tte.event['TIME'], float)
    ch = np.asarray(gbm_tte.event['PHA'], int) if 'PHA' in gbm_tte.event.dtype.names \
        else np.asarray(gbm_tte.event['CHANNEL'], int)
    energy = None
    if 'ENERGY' in gbm_tte.event.dtype.names:
        energy = np.asarray(gbm_tte.event['ENERGY'], float)

    t1, t2 = offset + ctx.t1, offset + ctx.t2
    off = ((ev >= offset - 20) & (ev < t1)) | ((ev > t2) & (ev <= t2 + 50))
    on = (ev >= t1) & (ev <= t2)
    t_on = t2 - t1
    t_off = 20.0 + 50.0

    base = tint_spec_base(ctx, det)
    spec = load_spec(base)
    nchan = len(spec['e'])
    # map PHA channel 0..nchan-1
    ch = np.clip(ch, 0, nchan - 1)
    on_c = np.bincount(ch[on], minlength=nchan).astype(float)
    off_c = np.bincount(ch[off], minlength=nchan).astype(float)
    off_rate = off_c / t_off
    heapy_bkg_rate = spec['bkg_rate']
    on_rate = on_c / t_on

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.step(spec['e'], on_rate / spec['de'], where='mid', label='on-pulse (TTE)', color='#1f77b4')
    ax.step(spec['e'], off_rate / spec['de'], where='mid', label='off-pulse mean (TTE)', color='#ff7f0e')
    ax.step(spec['e'], heapy_bkg_rate / spec['de'], where='mid', label='heapy .bkg', color='#d62728')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(8, 900)
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('counts s$^{-1}$ keV$^{-1}$')
    ax.set_title(f'{ctx.name} {det}  on [{ctx.t1},{ctx.t2}] vs off-pulse vs heapy bkg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(OUTDIR, f'{det}_bkg_compare.png')
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)

    ratio = heapy_bkg_rate / np.maximum(off_rate, 1e-12)
    m = (spec['e'] >= 8) & (spec['e'] <= 900) & (off_rate > 0) & np.isfinite(ratio)
    if np.any(m):
        print(f'  {det} heapy_bkg / offpulse median={np.median(ratio[m]):.3f}  '
              f'p16={np.percentile(ratio[m], 16):.3f}  p84={np.percentile(ratio[m], 84):.3f}')
        hi = m & (spec['e'] >= 200)
        if np.any(hi):
            print(f'  {det} E>200 keV  heapy/offpulse median={np.median(ratio[hi]):.3f}')
    else:
        print(f'  {det} off-pulse comparison: no overlapping channels (PHA vs TTE mapping)')
        print(f'    on-pulse counts={on_c.sum():.0f}  off-pulse counts={off_c.sum():.0f}  '
              f'heapy bkg counts={spec["bkg"].sum():.0f}')
    return energy


def print_cpl_def():
    path = '/usr/local/lib/python3.12/site-packages/bayspec/model/local/additive.py'
    print('\n=== bayspec cpl: vfv_peak=True uses nuFnu Ep, Ec=Ep/(2+alpha) ===')
    with open(path) as f:
        lines = f.readlines()
    print(''.join(lines[70:141]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    from pipeline.runner import GRBPipelineRunner
    from pipeline.spectra import active_heapy_tint_dir, active_heapy_tres_dir, tint_spec_base
    from pipeline.paths import spec_slice_name as _ssn

    runner = GRBPipelineRunner(n_workers=1, model_name='cpl', nlive=1000, force=False)
    ctx = runner.load('GRB140606B')
    print('dets', ctx.sel_dets, 'slices', ctx.time_slices)
    print('ext_fp via load printed above')

    print_cpl_def()

    z = float(ctx.z or 0.384)
    paper_rest = 352.0
    paper_obs = paper_rest / (1 + z)
    fermi_obs = 579.0
    fermi_rest = fermi_obs * (1 + z)
    print('\n=== literature Ep ===')
    print(f'  Yan+2024 Table 1 (Minaev 2020)  rest={paper_rest} keV  obs={paper_obs:.1f} keV')
    print(f'  Fermi GBM COMP catalog (Cano+2015) obs={fermi_obs} keV  rest={fermi_rest:.1f} keV')
    print(f'  Our previous tint CPL (n3,n4,n8 only) obs=581 keV  rest={581*(1+z):.1f} keV')

    tint_specs = []
    print('\n=== tint net S/N ===')
    for det in ctx.sel_dets:
        spec = load_spec(tint_spec_base(ctx, det))
        tint_specs.append((det, spec))
        if det[0] == 'n':
            bands = [(8, 50), (50, 300), (300, 900)]
        else:
            bands = [(300, 1000), (1000, 5000), (5000, 38000)]
        bits = []
        for lo, hi in bands:
            net, snr, src = band_snr(spec, lo, hi)
            bits.append(f'{lo}-{hi}: net={net:.0f} S/N={snr:.1f}')
        print(f'  {det}  tsrc={spec["tsrc"]:.2f}s  ' + '  '.join(bits))
    plot_count_spec(tint_specs, 'Time-integrated count spectra (heapy)', os.path.join(OUTDIR, 'tint_counts.png'))

    heapy = active_heapy_tres_dir(ctx)
    print('\n=== tres net S/N (n3 + b0) ===')
    rows = []
    for i, (t0, t1) in enumerate(ctx.time_slices, 1):
        name = _ssn(t0, t1)
        sdir = os.path.join(heapy, f'slice_{i:02d}')
        for det in ('n3', 'b0'):
            spec = load_spec(os.path.join(sdir, f'{name}_{det}'))
            if det == 'n3':
                net_lo, snr_lo, _ = band_snr(spec, 8, 50)
                net_mid, snr_mid, _ = band_snr(spec, 50, 300)
                net_hi, snr_hi, _ = band_snr(spec, 300, 900)
            else:
                net_lo, snr_lo, _ = band_snr(spec, 300, 1000)
                net_mid, snr_mid, _ = band_snr(spec, 1000, 5000)
                net_hi, snr_hi, _ = band_snr(spec, 5000, 38000)
            print(f'  slice{i:02d} {det} [{t0},{t1}]  '
                  f'lo S/N={snr_lo:.1f}  mid S/N={snr_mid:.1f}  hi S/N={snr_hi:.1f}')
            rows.append((i, det, spec))
        if i in (1, 4, 6):
            specs = [(f'slice {i} {d}', load_spec(os.path.join(sdir, f'{name}_{d}')))
                     for d in ctx.sel_dets]
            plot_count_spec(specs, f'Slice {i} [{t0},{t1}] s', os.path.join(OUTDIR, f'slice_{i:02d}_counts.png'))

    print('\n=== heapy bkg vs off-pulse TTE (paper-like channel baseline) ===')
    offpulse_vs_heapy(ctx, 'n3')
    offpulse_vs_heapy(ctx, 'b0')

    # existing tint data.json dets
    dj = '/workspace/data/GRB140606B/data/tintegrated/bayspec/cpl/data.json'
    if os.path.isfile(dj):
        data = json.load(open(dj))
        print('\n=== existing tint fit detectors ===', [d['Name'] for d in data])


if __name__ == '__main__':
    main()
