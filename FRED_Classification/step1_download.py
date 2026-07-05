#!/usr/bin/env python3
"""
Step 1: Download Fermi GBM burst catalog and CTIME lightcurve files.

Run:
    python step1_download.py

Outputs (in ./grb_data/):
    manifest.json          – list of downloaded bursts + file paths
    fits/ctime_*.pha       – CTIME FITS files (lightcurve data)
    plots/*_lc.png         – quicklook lightcurve PNGs
"""

import os, io, json, time, requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
N_BURSTS = 50         # bursts to download (start small; scale to thousands later)
MIN_T90  = 2.0        # seconds — keep long GRBs only
OUTDIR   = '/workspace/data/grb_data'

os.makedirs(f'{OUTDIR}/fits',  exist_ok=True)
os.makedirs(f'{OUTDIR}/plots', exist_ok=True)

# NaI detector labels (try in this order until one downloads)
NAI_DETS = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CATALOG QUERY via HEASARC TAP
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_catalog(n_request=500, min_t90=2.0):
    """
    Query the HEASARC FERMIGBRST catalog via TAP VOTable.
    Falls back to a built-in sample if network fails.
    """
    # ── Primary: TAP VOTable ───────────────────────────────────────────────
    try:
        print("Querying HEASARC Fermi GBM Burst Catalog (TAP)...")
        url = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
        query = (f"SELECT TOP {n_request} "
                 f"trigger_name, t90, fluence, flux_1024, bcat_detector_mask "
                 f"FROM fermigbrst "
                 f"WHERE t90 > {min_t90} "
                 f"ORDER BY trigger_name")
        r = requests.get(url,
                         params={'QUERY': query, 'FORMAT': 'votable', 'LANG': 'ADQL'},
                         timeout=120)
        if r.status_code == 200:
            cat = Table.read(io.BytesIO(r.content), format='votable')
            # Normalize column names to lowercase
            for col in cat.colnames:
                cat.rename_column(col, col.lower().strip())
            print(f"  → {len(cat)} long GRBs retrieved")
            return cat
    except Exception as e:
        print(f"  TAP failed: {e}")

    # ── Fallback: built-in sample of 27 well-known long GRBs ──────────────
    print("  ⚠️  HEASARC not reachable — using built-in 27-burst sample")
    sample = [
        ('bn080916009',  66.56, 2.09e-04, 1.22e-06, '110000001100'),
        ('bn090902462',  21.60, 4.41e-04, 2.73e-06, '011000000110'),
        ('bn091208410',  27.84, 3.22e-05, 3.03e-07, '000011100000'),
        ('bn100814160',  21.76, 2.46e-05, 6.10e-07, '001100000000'),
        ('bn110213220',  48.38, 2.17e-05, 4.89e-07, '100001000000'),
        ('bn110920546',  47.10, 3.08e-05, 4.51e-07, '010100000000'),
        ('bn120328268',  71.17, 3.56e-05, 5.09e-07, '001100000000'),
        ('bn120624933', 270.97, 7.51e-05, 4.57e-07, '010001000000'),
        ('bn130131350',  35.65, 3.94e-05, 4.96e-07, '100000100000'),
        ('bn131011741',  39.17, 3.39e-04, 2.16e-06, '110000001100'),
        ('bn140606133',  27.14, 1.47e-04, 2.38e-06, '001100000000'),
        ('bn150514774',  13.57, 3.72e-05, 8.11e-07, '110000001100'),
        ('bn151027166',  89.60, 1.42e-04, 8.56e-07, '000001010000'),
        ('bn160408060',  25.09, 2.96e-05, 6.11e-07, '010100000000'),
        ('bn160625945', 453.00, 3.95e-04, 1.78e-06, '110000001100'),
        ('bn161129300',  14.59, 4.47e-05, 1.14e-06, '001010000000'),
        ('bn170207813',  26.55, 3.04e-05, 5.35e-07, '000100100000'),
        ('bn171010792', 107.32, 1.09e-03, 5.07e-06, '011000000110'),
        ('bn180120815',  33.54, 3.28e-05, 5.01e-07, '100001000000'),
        ('bn180703876',  14.59, 2.30e-05, 4.57e-07, '001100000000'),
        ('bn190114873',  61.44, 1.72e-03, 2.60e-05, '001001000000'),
        ('bn190829672',  30.72, 4.79e-04, 4.47e-06, '110000001100'),
        ('bn210204143',  49.15, 1.93e-05, 3.74e-07, '001001000000'),
        ('bn220101215',  22.53, 2.64e-05, 5.88e-07, '100000100000'),
        ('bn170607596',  21.50, 1.83e-05, 4.12e-07, '001100000000'),
        ('bn120326056',  69.89, 3.97e-05, 5.44e-07, '001100000000'),
        ('bn110920546',  47.10, 3.08e-05, 4.51e-07, '010100000000'),
    ]
    cat = Table(rows=sample,
                names=['trigger_name', 't90', 'fluence',
                       'flux_1024', 'bcat_detector_mask'])
    print(f"  → {len(cat)} bursts in built-in sample")
    return cat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOWNLOAD CTIME FITS FILE
# ═══════════════════════════════════════════════════════════════════════════════
def trigger_year(trigger_name: str) -> str:
    """bn140606133 → '2014'"""
    return f"20{trigger_name[2:4]}"


def download_ctime(trigger_name: str, outdir: str):
    """
    Download one CTIME PHA file for a GRB trigger from HEASARC.
    Tries NaI detectors n0–n3 and file versions v00–v02 until one succeeds.

    Returns (local_filepath, detector_name) or (None, None) on failure.
    """
    year = trigger_year(trigger_name)
    base = (f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/"
            f"triggers/{year}/{trigger_name}/current")

    for det in NAI_DETS[:6]:          # try first 6 NaI detectors
        for ver in ['v00', 'v01', 'v02']:
            url = f"{base}/glg_ctime_{det}_{trigger_name}_{ver}.pha"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    fpath = os.path.join(outdir, 'fits',
                                         f"ctime_{trigger_name}_{det}.pha")
                    with open(fpath, 'wb') as f:
                        f.write(r.content)
                    return fpath, det
            except requests.RequestException:
                continue
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. READ CTIME → (time_array, rate_array)
# ═══════════════════════════════════════════════════════════════════════════════
def read_ctime(fpath: str):
    """
    Parse a GBM CTIME FITS file.
    Returns (time_centers_s, count_rate, trigger_time_met).

    CTIME has 8 energy channels. We sum channels 1–7 (avoids the
    noisy lowest channel) and divide by exposure to get count rate.
    """
    with fits.open(fpath) as hdul:
        spec     = hdul['SPECTRUM'].data
        trigtime = hdul[0].header.get('TRIGTIME', 0.0)

        t_start  = spec['TIME']     - trigtime   # relative to trigger
        t_end    = spec['ENDTIME']  - trigtime
        counts   = spec['COUNTS']                # shape (N_bins, 8)
        exposure = spec['EXPOSURE']

    t_centers = 0.5 * (t_start + t_end)
    rate      = counts[:, 1:].sum(axis=1) / np.clip(exposure, 1e-4, None)
    return t_centers, rate, trigtime


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOT LIGHTCURVE PNG
# ═══════════════════════════════════════════════════════════════════════════════
def plot_lightcurve(trigger_name: str, t: np.ndarray, rate: np.ndarray,
                    t90: float, outdir: str) -> str:
    """Save a dark-themed lightcurve PNG. Returns file path."""
    BG, FG, ACC = '#0F2024', '#EFE7D8', '#EF7B45'

    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG)
    ax.set_facecolor(BG)

    ax.step(t, rate, where='mid', color=ACC, lw=1.5)
    ax.fill_between(t, 0, rate, step='mid', color=ACC, alpha=0.20)
    ax.axvline(0, color='#F0C987', lw=1.0, ls='--', alpha=0.7, label='trigger')

    ax.set_xlabel('Time since trigger (s)', color=FG, fontsize=9)
    ax.set_ylabel('Count rate (ct/s)', color=FG, fontsize=9)
    ax.set_title(f'{trigger_name}   T90 = {t90:.1f} s', color=FG,
                 fontsize=10, fontweight='bold')

    ax.tick_params(colors='#B8A88F', labelsize=8)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#B8A88F')

    plt.tight_layout()
    fpath = os.path.join(outdir, 'plots', f'{trigger_name}_lc.png')
    plt.savefig(fpath, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()
    return fpath


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN DOWNLOAD LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Request 3× what we need to account for download failures
    catalog = fetch_catalog(n_request=N_BURSTS * 3, min_t90=MIN_T90)

    downloaded = []
    pbar = tqdm(catalog, desc='Downloading bursts', unit='grb')

    for row in pbar:
        trigger = row['trigger_name'].strip()
        pbar.set_postfix(trigger=trigger, ok=len(downloaded))

        # ── Download CTIME ────────────────────────────────────────────────────
        fpath, det = download_ctime(trigger, OUTDIR)
        if fpath is None:
            continue

        # ── Read and quality-check ────────────────────────────────────────────
        try:
            t, rate, _ = read_ctime(fpath)
        except Exception as e:
            os.remove(fpath)
            continue

        # Keep only ±200 s around trigger; require at least 20 bins
        mask = (t > -60) & (t < 200)
        if mask.sum() < 20:
            continue

        # ── Plot PNG ──────────────────────────────────────────────────────────
        t90 = float(row['t90']) if row['t90'] is not None else 0.0
        png = plot_lightcurve(trigger, t[mask], rate[mask], t90, OUTDIR)

        downloaded.append({
            'trigger_name': trigger,
            'fits':         fpath,
            'png':          png,
            'det':          det,
            't90':          t90,
            'fluence':      float(row['fluence']) if row['fluence'] is not None else None,
            'flux_1024':    float(row['flux_1024']) if row['flux_1024'] is not None else None,
        })

        time.sleep(0.15)           # polite rate limit to HEASARC
        if len(downloaded) >= N_BURSTS:
            break

    pbar.close()

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest_path = f'{OUTDIR}/manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(downloaded, f, indent=2)

    print(f"\n{'─'*55}")
    print(f"  Downloaded : {len(downloaded)} bursts")
    print(f"  FITS files : {OUTDIR}/fits/")
    print(f"  PNG plots  : {OUTDIR}/plots/")
    print(f"  Manifest   : {manifest_path}")
    print(f"{'─'*55}")
    print("Next: open data/grb_data/plots/ and label bursts,")
    print("      then run:  python step2_features_label.py")


if __name__ == '__main__':
    main()