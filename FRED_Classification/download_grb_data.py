#!/usr/bin/env python3
"""
download_grb_data.py — Download GBM burst catalog + CTIME lightcurve files
using the official NASA gdt-fermi library (astro-gdt-fermi).

Run cell by cell in Jupyter.
"""

import os, time
import pandas as pd
from gdt.missions.fermi.gbm.catalogs import BurstCatalog
from gdt.missions.fermi.gbm.finders import TriggerFtp

OUTDIR = '/workspace/data/grb_data'
os.makedirs(f'{OUTDIR}/fits', exist_ok=True)


# ── CELL 1: Fetch burst catalog ───────────────────────────────────────────────
def fetch_catalog(max_retries=3):
    """
    Downloads the full GBM Burst Catalog from HEASARC (~75s first time).
    Retries on connection drops (common with this large 306-column table).
    Returns a BurstCatalog object — use .get_table() to see it as a table.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading burst catalog from HEASARC "
                  f"(attempt {attempt}/{max_retries}, may take ~1 min)...")
            cat = BurstCatalog()
            print(f"  → {cat}")
            return cat
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(5)
    raise RuntimeError("Could not download catalog after "
                       f"{max_retries} attempts. Try again later — "
                       "HEASARC connection may be unstable right now.")


def filter_long_grbs(cat, min_t90=2.0):
    """
    Filter catalog to long GRBs only. Returns astropy Table.
    Bypasses cat.slice() (unreliable in this library version) and
    filters manually with pandas instead.
    """
    print(f"  Available columns (first 20): {cat.columns[:20]}")

    # Catalog uses 'NAME' not 'trigger_name', and 'T90' uppercase
    name_col = 'name' if 'NAME' in cat.columns else 'trigger_name'
    t90_col  = 'T90'

    table = cat.get_table(columns=(name_col, t90_col, 'FLUENCE', 'FLUX_1024'))
    df = pd.DataFrame(table)
    df.columns = ['trigger_name', 't90', 'fluence', 'flux_1024']

    df = df.dropna(subset=['t90'])
    df = df[df['t90'] > min_t90].reset_index(drop=True)

    print(f"  → {len(df)} long GRBs (T90 > {min_t90}s) out of {len(table)} total")
    return df


# ── CELL 2: Download CTIME files ──────────────────────────────────────────────
def download_ctime(trigger_name, outdir):
    """
    Download CTIME files for one trigger using the official finder.
    trigger_name format: 'bn140606133' or just '140606133'
    Returns list of downloaded file paths.
    """
    trig_id = trigger_name.replace('bn', '').replace('GRB', '')
    try:
        finder = TriggerFtp(trig_id)
        finder.get_ctime(outdir)
        files = [f for f in os.listdir(outdir) if trig_id in f and 'ctime' in f]
        return files
    except Exception as e:
        print(f"  {trigger_name}: {e}")
        return []


def download_all(table, n_bursts, outdir):
    """Download CTIME files for N bursts from the catalog DataFrame."""
    downloaded = []
    for _, row in table.iloc[:n_bursts * 2].iterrows():   # buffer for failures
        trigger = str(row['trigger_name']).strip()
        files = download_ctime(trigger, f'{outdir}/fits')
        if files:
            downloaded.append({'trigger_name': trigger,
                               't90': float(row['t90']),
                               'files': files})
            print(f"  ✓ {trigger}  ({len(files)} files)")
        if len(downloaded) >= n_bursts:
            break

    print(f"\nDownloaded {len(downloaded)} bursts → {outdir}/fits/")
    return downloaded


# ── USAGE ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cat   = fetch_catalog()
    table = filter_long_grbs(cat, min_t90=2.0)
    results = download_all(table, n_bursts=20, outdir=OUTDIR)
