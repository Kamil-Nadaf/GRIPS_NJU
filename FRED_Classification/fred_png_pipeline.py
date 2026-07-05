#!/usr/bin/env python3
"""
FRED Classifier — clean notebook pipeline
Run each section as a separate cell.
"""

import os, io, json, time, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.table import Table
from astropy.io import fits
from tqdm.notebook import tqdm

OUTDIR = '/workspace/data/grb_data'
os.makedirs(f'{OUTDIR}/plots', exist_ok=True)
os.makedirs(f'{OUTDIR}/fits',  exist_ok=True)

NAI_DETS = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']


# ── CELL 1: Fetch catalog ─────────────────────────────────────────────────────
def fetch_catalog(n=150, min_t90=2.0):
    print("Querying HEASARC catalog...")
    url   = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
    query = (f"SELECT TOP {n} trigger_name, t90, fluence, flux_1024 "
             f"FROM fermigbrst WHERE t90 > {min_t90} ORDER BY trigger_name")

    for attempt in range(2):
        try:
            r = requests.get(url, params={'QUERY': query, 'FORMAT': 'votable',
                                          'LANG': 'ADQL'}, timeout=45)
            if r.status_code == 200:
                cat = Table.read(io.BytesIO(r.content), format='votable')
                for col in cat.colnames:
                    cat.rename_column(col, col.lower().strip())
                df = cat.to_pandas()
                print(f"  → {len(df)} bursts")
                return df
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")

    print("  ⚠ HEASARC slow/unreachable — using built-in 25-burst sample")
    rows = [
        ('bn080916009', 66.56, 2.09e-4, 1.22e-6),
        ('bn090902462', 21.60, 4.41e-4, 2.73e-6),
        ('bn091208410', 27.84, 3.22e-5, 3.03e-7),
        ('bn100814160', 21.76, 2.46e-5, 6.10e-7),
        ('bn110213220', 48.38, 2.17e-5, 4.89e-7),
        ('bn110920546', 47.10, 3.08e-5, 4.51e-7),
        ('bn120328268', 71.17, 3.56e-5, 5.09e-7),
        ('bn120624933',270.97, 7.51e-5, 4.57e-7),
        ('bn130131350', 35.65, 3.94e-5, 4.96e-7),
        ('bn131011741', 39.17, 3.39e-4, 2.16e-6),
        ('bn140606133', 27.14, 1.47e-4, 2.38e-6),
        ('bn150514774', 13.57, 3.72e-5, 8.11e-7),
        ('bn151027166', 89.60, 1.42e-4, 8.56e-7),
        ('bn160408060', 25.09, 2.96e-5, 6.11e-7),
        ('bn160625945',453.00, 3.95e-4, 1.78e-6),
        ('bn161129300', 14.59, 4.47e-5, 1.14e-6),
        ('bn170207813', 26.55, 3.04e-5, 5.35e-7),
        ('bn171010792',107.32, 1.09e-3, 5.07e-6),
        ('bn180120815', 33.54, 3.28e-5, 5.01e-7),
        ('bn190114873', 61.44, 1.72e-3, 2.60e-5),
        ('bn190829672', 30.72, 4.79e-4, 4.47e-6),
        ('bn210204143', 49.15, 1.93e-5, 3.74e-7),
        ('bn220101215', 22.53, 2.64e-5, 5.88e-7),
        ('bn170607596', 21.50, 1.83e-5, 4.12e-7),
        ('bn120326056', 69.89, 3.97e-5, 5.44e-7),
    ]
    df = pd.DataFrame(rows, columns=['trigger_name','t90','fluence','flux_1024'])
    print(f"  → {len(df)} bursts in built-in sample")
    return df


# ── CELL 2: Download CTIME + generate PNG ────────────────────────────────────
def trigger_year(name): return f"20{name[2:4]}"

def download_and_plot(trigger, t90, outdir):
    """Download one CTIME file, generate lightcurve PNG. Returns png path or None."""
    year = trigger_year(trigger)
    base = (f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/"
            f"triggers/{year}/{trigger}/current")

    # Try detectors until one downloads
    fpath = None
    for det in NAI_DETS[:6]:
        for ver in ['v00','v01','v02']:
            url = f"{base}/glg_ctime_{det}_{trigger}_{ver}.pha"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    fpath = f"{outdir}/fits/{trigger}.pha"
                    with open(fpath, 'wb') as f:
                        f.write(r.content)
                    break
            except: continue
        if fpath: break

    if fpath is None:
        return None

    # Read and plot
    try:
        with fits.open(fpath) as hdul:
            spec     = hdul['SPECTRUM'].data
            trigtime = hdul[0].header.get('TRIGTIME', 0.0)
            t        = 0.5*(spec['TIME'] + spec['ENDTIME']) - trigtime
            counts   = spec['COUNTS'][:,1:].sum(axis=1)
            exposure = spec['EXPOSURE']
        rate = counts / np.clip(exposure, 1e-4, None)
        mask = (t > -30) & (t < max(t90*2, 60))

        fig, ax = plt.subplots(figsize=(7, 2.8), facecolor='#0F2024')
        ax.set_facecolor('#0F2024')
        ax.step(t[mask], rate[mask], where='mid', color='#EF7B45', lw=1.2)
        ax.fill_between(t[mask], 0, rate[mask], step='mid', color='#EF7B45', alpha=0.2)
        ax.axvline(0, color='#F0C987', lw=0.8, ls='--', alpha=0.6)
        ax.set_xlabel('Time since trigger (s)', color='#EFE7D8', fontsize=8)
        ax.set_ylabel('ct/s', color='#EFE7D8', fontsize=8)
        ax.tick_params(colors='#B8A88F', labelsize=7)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax.spines[sp].set_color('#B8A88F')
        plt.tight_layout()

        png = f"{outdir}/plots/{trigger}_lc.png"
        plt.savefig(png, dpi=100, bbox_inches='tight', facecolor='#0F2024')
        plt.close()
        return png
    except Exception as e:
        plt.close()
        return None


def download_all(catalog_df, n_bursts, outdir):
    results = []
    pbar = tqdm(catalog_df.iterrows(), total=len(catalog_df), desc='Downloading')

    for _, row in pbar:
        trigger = str(row['trigger_name']).strip()
        t90     = float(row.get('t90') or 0)
        pbar.set_postfix(ok=len(results), trigger=trigger)

        png = download_and_plot(trigger, t90, outdir)
        if png is None:
            continue

        results.append({
            'trigger_name': trigger,
            'png':          png,
            't90':          t90,
            'fluence':      float(row.get('fluence') or 0),
            'flux_1024':    float(row.get('flux_1024') or 0),
        })
        time.sleep(0.1)
        if len(results) >= n_bursts:
            break

    df = pd.DataFrame(results)
    df.to_csv(f'{outdir}/downloaded.csv', index=False)
    print(f"\nDownloaded {len(df)} lightcurves → {outdir}/plots/")
    return df


# ── CELL 3: Browse PNGs in notebook ──────────────────────────────────────────
def show_page(df, page=0, per_page=9):
    if len(df) == 0:
        print("No bursts downloaded yet.")
        return
    subset = df.iloc[page*per_page : (page+1)*per_page]
    n      = len(subset)
    if n == 0:
        print(f"Page {page} is empty. Max page = {len(df)//per_page}")
        return

    cols = 3
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*3.2))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(subset.iterrows()):
        ax = axes[i]
        png = row.get('png','')
        if png and os.path.exists(str(png)):
            img = mpimg.imread(str(png))
            ax.imshow(img, aspect='auto')
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                    transform=ax.transAxes)
        ax.set_title(f"[{page*per_page+i}] {row['trigger_name']}\n"
                     f"T90={row['t90']:.1f}s", fontsize=8, fontweight='bold')
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    plt.suptitle(f"Page {page}  (indices {page*per_page}–{page*per_page+n-1})  "
                 f"| Total pages: 0–{len(df)//per_page}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/page_{page}.png', dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Tip: label FRED bursts by index shown above (e.g. 0,2,5)")


# ── CELL 4: Label ─────────────────────────────────────────────────────────────
def label_from_page(df, fred_indices, page, label_file, per_page=9):
    """
    Label one page at a time.
    fred_indices : list of global indices that are FRED on this page
                   e.g. [0, 3, 7]

    Example usage (after show_page(df, page=0)):
        label_from_page(df, fred_indices=[0,3], page=0, label_file=LABEL_FILE)
    """
    labels = {}
    if os.path.exists(label_file):
        with open(label_file) as f:
            labels = json.load(f)

    start = page * per_page
    end   = min(start + per_page, len(df))
    fred_set = set(fred_indices)

    for idx in range(start, end):
        trigger = df.iloc[idx]['trigger_name']
        labels[trigger] = 1 if idx in fred_set else 0

    with open(label_file, 'w') as f:
        json.dump(labels, f, indent=2)

    n_fred = sum(v == 1 for v in labels.values())
    print(f"Saved. Total labeled: {len(labels)}  "
          f"(FRED={n_fred}, non-FRED={len(labels)-n_fred})")
    return labels


# ── CELL 5: Train ─────────────────────────────────────────────────────────────
def train_classifier(df, labels, outdir):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import joblib

    records = []
    for trigger, lbl in labels.items():
        row = df[df['trigger_name'] == trigger]
        if row.empty: continue
        row = row.iloc[0]
        t90     = float(row['t90'])
        fluence = float(row['fluence'])
        flux    = float(row['flux_1024'])
        records.append({
            'log_t90':     np.log10(max(t90, 0.01)),
            'log_fluence': np.log10(max(fluence, 1e-10)),
            'log_flux':    np.log10(max(flux, 1e-10)),
            'hardness':    flux / max(fluence, 1e-10),
            'label':       lbl,
            'trigger_name': trigger,
        })

    feat_df  = pd.DataFrame(records).dropna()
    FEATURES = ['log_t90','log_fluence','log_flux','hardness']
    X = feat_df[FEATURES].values
    y = feat_df['label'].values.astype(int)

    print(f"Dataset: {len(feat_df)} bursts  "
          f"(FRED={y.sum()}, non-FRED={len(y)-y.sum()})")

    if y.sum() < 3 or (len(y)-y.sum()) < 3:
        print("Need at least 3 of each class. Label more pages first.")
        return None, feat_df

    n_splits = min(5, int(y.sum()), int(len(y)-y.sum()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        'Random Forest':      Pipeline([('sc', StandardScaler()),
                                        ('clf', RandomForestClassifier(
                                            n_estimators=200,
                                            class_weight='balanced',
                                            random_state=42))]),
        'Logistic Regression':Pipeline([('sc', StandardScaler()),
                                        ('clf', LogisticRegression(
                                            class_weight='balanced',
                                            max_iter=500))]),
    }

    print(f"\n── {n_splits}-fold CV ─────────────────────────────")
    best_model, best_score, best_name = None, 0, ''
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        print(f"  {name:22s}  F1 = {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_score:
            best_score, best_model, best_name = scores.mean(), model, name

    best_model.fit(X, y)
    feat_df['fred_prob'] = best_model.predict_proba(X)[:, 1]
    feat_df['fred_pred'] = best_model.predict(X)

    print(f"\n── {best_name} ─────────────────────────────────────")
    print(classification_report(y, feat_df['fred_pred'],
                                 target_names=['non-FRED','FRED']))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cm = confusion_matrix(y, feat_df['fred_pred'])
    ConfusionMatrixDisplay(cm, display_labels=['non-FRED','FRED']).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(f'{best_name}\nConfusion Matrix')

    clf = best_model.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
        idx = np.argsort(imp)
        axes[1].barh([FEATURES[i] for i in idx], imp[idx], color='#EF7B45')
        axes[1].set_title('Feature importance')
    plt.tight_layout()
    plt.savefig(f'{outdir}/classifier_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    joblib.dump({'model': best_model, 'features': FEATURES, 'name': best_name},
                f'{outdir}/fred_classifier.pkl')
    feat_df.to_csv(f'{outdir}/predictions.csv', index=False)
    print(f"\nModel → {outdir}/fred_classifier.pkl")
    return best_model, feat_df


LABEL_FILE = f'{OUTDIR}/labels.json'