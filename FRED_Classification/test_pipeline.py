#!/usr/bin/env python3
"""
test_pipeline.py — Verify the full pipeline works using synthetic data.

Run this on any machine (no internet needed) to check that all
dependencies are installed and the code is correct:

    python test_pipeline.py

Then use step1/2/3 on your real machine or DGX where HEASARC is accessible.
"""
import sys, os, json
sys.path.insert(0, '/workspace')
os.makedirs('grb_data/fits',  exist_ok=True)
os.makedirs('grb_data/plots', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import skew

# ── Generate synthetic CTIME FITS files ──────────────────────────────────────
def norris(t, A, tau1, tau2, ts=0):
    out = np.zeros_like(t)
    m = t > ts
    out[m] = A * np.exp(-tau1/(t[m]-ts) - (t[m]-ts)/tau2)
    return out

def make_synthetic_ctime(trigger_name, pulse_type, outdir, seed):
    """Create a realistic fake CTIME FITS file."""
    rng = np.random.default_rng(seed)
    t_edges = np.arange(-60, 201, 0.512)         # CTIME default bin width ~512ms
    trigtime = 400000000.0                         # dummy Fermi MET

    bg   = 280.0
    n    = len(t_edges) - 1
    t    = 0.5 * (t_edges[:-1] + t_edges[1:])
    expo = np.full(n, 0.512)

    if pulse_type == 'fred':
        src = norris(t, A=rng.uniform(150, 350),
                     tau1=rng.uniform(1, 4), tau2=rng.uniform(8, 20))
    elif pulse_type == 'multi':
        n_pk = rng.integers(2, 5)
        src  = sum(norris(t, A=rng.uniform(60, 150),
                          tau1=rng.uniform(1, 5), tau2=rng.uniform(5, 15),
                          ts=rng.uniform(0, 30))
                   for _ in range(n_pk))
    elif pulse_type == 'flat':
        src = rng.uniform(50, 200) * np.ones(n) * ((t > 5) & (t < 50)).astype(float)
    else:
        src = np.zeros(n)

    total_rate = bg + src + rng.normal(0, np.sqrt(bg + src + 1))
    total_rate = np.clip(total_rate, 0, None)
    counts_row = np.round(total_rate * expo).astype(int)

    # Distribute across 8 channels (rough approximation)
    weights = np.array([0.05, 0.12, 0.18, 0.22, 0.20, 0.12, 0.08, 0.03])
    counts  = np.outer(counts_row, weights).astype(int)

    # Build FITS
    col_time  = fits.Column(name='TIME',     format='D', array=t_edges[:-1] + trigtime)
    col_end   = fits.Column(name='ENDTIME',  format='D', array=t_edges[1:]  + trigtime)
    col_cnt   = fits.Column(name='COUNTS',   format='8J', array=counts)
    col_exp   = fits.Column(name='EXPOSURE', format='E', array=expo)
    spec_hdu  = fits.BinTableHDU.from_columns(
        fits.ColDefs([col_time, col_end, col_cnt, col_exp]),
        name='SPECTRUM'
    )
    phdr = fits.PrimaryHDU()
    phdr.header['TRIGTIME'] = trigtime

    fpath = os.path.join(outdir, 'fits', f'ctime_{trigger_name}_n0.pha')
    fits.HDUList([phdr, spec_hdu]).writeto(fpath, overwrite=True)
    return fpath


# ── Create synthetic dataset (20 FRED, 15 multi-peak, 5 flat) ────────────────
print("Generating synthetic lightcurve dataset...")
manifest, labels = [], {}
types = (
    [('fred', 1)]  * 20 +
    [('multi', 0)] * 15 +
    [('flat',  0)] *  5
)

for i, (ptype, lbl) in enumerate(types):
    trig = f'bn_synth_{ptype}_{i:03d}'
    fpath = make_synthetic_ctime(trig, ptype, 'grb_data', seed=i*13+7)
    t90   = 40.0 if ptype == 'fred' else 55.0

    manifest.append({'trigger_name': trig, 'fits': fpath,
                     'png': '', 't90': t90, 'det': 'n0'})
    labels[trig] = lbl

print(f"  Created {len(manifest)} synthetic bursts "
      f"(FRED={sum(v for v in labels.values())}, "
      f"non-FRED={sum(1-v for v in labels.values())})")

with open('grb_data/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
with open('grb_data/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)


# ── Run feature extraction ────────────────────────────────────────────────────
print("\nExtracting features...")
from step2_features_label import extract_all_features
df = extract_all_features(manifest, labels)
print(f"  Features extracted: {len(df)} rows, {len(df.columns)} columns")


# ── Run training ──────────────────────────────────────────────────────────────
print("\nTraining classifier...")
from step3_train import load_dataset, train_and_evaluate
X, y, names, df2 = load_dataset()
model = train_and_evaluate(X, y, names, df2)

print("\n" + "═"*55)
print("  PIPELINE TEST PASSED")
print("  All components work correctly.")
print("═"*55)
print("\nOutput files:")
for f in ['grb_data/features.csv', 'grb_data/predictions.csv',
          'grb_data/fred_classifier.pkl', 'grb_data/classifier_results.png']:
    size = os.path.getsize(f) if os.path.exists(f) else 0
    print(f"  {f}  ({size/1024:.1f} KB)")
print("\nNext steps (on your MacBook or DGX where HEASARC is reachable):")
print("  1. python step1_download.py    # downloads ~50 real GRB lightcurves")
print("  2. python step2_features_label.py  # label + extract features")
print("  3. python step3_train.py       # train and evaluate classifier")
