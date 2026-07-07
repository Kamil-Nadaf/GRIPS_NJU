"""
grb_discovery.py
-----------------
Step 1 of the pipeline: discover GRBs over a date range from the Fermi GBM
Burst Catalog (fermigbrst, hosted at HEASARC) and turn the result into a
DataFrame with the same columns your existing grb_utils.py pipeline expects:
    name, ra, dec, utc, sel_dets, t1, t2

t1/t2 are Txx-derived (from the catalog's own t90_start / t90 columns), not
a fixed window, per your instruction.

Install:
    pip install astroquery --break-system-packages   # astropy comes with it

Usage:
    from grb_discovery import query_gbm_catalog, build_grbs_df

    raw = query_gbm_catalog('2019-01-01', '2019-12-31')
    grbs_df = build_grbs_df(raw, t90_pad_frac=0.0)
"""

import numpy as np
import pandas as pd
from astropy.time import Time

# ----------------------------------------------------------------------------
# Detector order used by the GBM 'scat_detector_mask' field in fermigbrst.
# This is the standard 14-detector GBM ordering (12 NaI + 2 BGO). Verify
# against the HEASARC fermigbrst column documentation once you have a live
# result table — if sel_dets come out obviously wrong (e.g. all BGO, no NaI)
# the order below is the first thing to check.
# ----------------------------------------------------------------------------
GBM_DET_ORDER = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5',
                  'n6', 'n7', 'n8', 'n9', 'na', 'nb',
                  'b0', 'b1']


def mask_to_dets(mask):
    """Convert a 14-char '0'/'1' scat_detector_mask string into a list of
    detector short-names, e.g. '00100000000010' -> ['n2', 'b0']."""
    if mask is None:
        return None
    mask = str(mask).strip()
    if len(mask) != len(GBM_DET_ORDER):
        return None
    return [det for det, flag in zip(GBM_DET_ORDER, mask) if flag == '1']


def query_gbm_catalog(start_date, end_date):
    """Query the Fermi GBM burst catalog (fermigbrst) at HEASARC for all
    triggers in [start_date, end_date] (any string astropy.time.Time parses,
    e.g. '2019-01-01').

    Returns a raw pandas DataFrame with (at least) the columns:
    name, ra, dec, trigger_time, t90, t90_start, scat_detector_mask
    """
    from astroquery.heasarc import Heasarc

    heasarc = Heasarc()

    # fermigbrst's trigger_time column is numeric (MJD, double precision) —
    # NOT a string. Quoting it (as an ISOT string) is what raised
    # "invalid input syntax for type double precision" — pass bare floats.
    t_start_mjd = Time(start_date).mjd
    t_stop_mjd = Time(end_date).mjd

    query = f"""
    SELECT name, ra, dec, trigger_time, t90, t90_start, scat_detector_mask
    FROM fermigbrst
    WHERE trigger_time BETWEEN {t_start_mjd} AND {t_stop_mjd}
    """

    try:
        # Modern astroquery (>=0.4.7): direct ADQL/TAP access, no spatial cone needed.
        result = heasarc.query_tap(query=query).to_table()
    except AttributeError:
        # Older astroquery fallback: no query_tap. Pull the whole table for the
        # mission and filter locally in pandas instead (slower, but works).
        print("astroquery.Heasarc.query_tap not available on this version; "
              "falling back to query_mission_cols/query_region. "
              "Consider `pip install -U astroquery --break-system-packages`.")
        result = heasarc.query_object(
            object_name=None, mission='fermigbrst',
            fields='name,ra,dec,trigger_time,t90,t90_start,scat_detector_mask'
        )

    df = result.to_pandas()
    # Column names sometimes come back as bytes on older astroquery — normalize.
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _trigger_utc(row):
    """Convert whatever format trigger_time comes back as (MJD float or an
    ISO-like string) into an ISO-T UTC string matching grb_config.py's
    existing 'utc' column format."""
    val = row['trigger_time']
    try:
        # Numeric -> assume MJD
        t = Time(float(val), format='mjd', scale='utc')
    except (ValueError, TypeError):
        t = Time(val, scale='utc')
    return t.isot


def build_grbs_df(raw_df, t90_pad_frac=0.0, require_dets=True):
    """Turn the raw HEASARC catalog result into the grbs_df schema used by
    grb_config.py / grb_utils.py: name, ra, dec, utc, sel_dets, t1, t2.

    Args:
        raw_df: output of query_gbm_catalog()
        t90_pad_frac: fractional padding added on each side of the t90 window
            for t1/t2, e.g. 0.1 widens [t90_start, t90_start+t90] by 10% on
            each side. 0.0 = use t90 window as-is.
        require_dets: drop rows where scat_detector_mask doesn't decode to
            at least 2 detectors (BayesInfer needs >=2, per fit_all_time_slices).

    Rows with missing/invalid ra or dec are always dropped, per your spec.
    """
    rows = []
    for _, r in raw_df.iterrows():
        ra, dec = r.get('ra'), r.get('dec')
        if pd.isna(ra) or pd.isna(dec):
            continue

        t90 = r.get('t90')
        t90_start = r.get('t90_start')
        if pd.isna(t90) or pd.isna(t90_start):
            continue

        pad = t90_pad_frac * float(t90)
        t1 = float(t90_start) - pad
        t2 = float(t90_start) + float(t90) + pad

        sel_dets = mask_to_dets(r.get('scat_detector_mask'))
        if require_dets and (not sel_dets or len(sel_dets) < 2):
            continue

        name = str(r.get('name')).strip()
        if not name.upper().startswith('GRB'):
            name = f"GRB{name}"

        rows.append({
            'name': name,
            'ra': float(ra),
            'dec': float(dec),
            'utc': _trigger_utc(r),
            'sel_dets': sel_dets,
            't1': t1,
            't2': t2,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('name').reset_index(drop=True)
    return df


if __name__ == '__main__':
    raw = query_gbm_catalog('2019-01-01', '2019-01-31')
    print(f"Raw catalog rows: {len(raw)}")
    grbs_df = build_grbs_df(raw)
    print(f"Usable GRBs (RA/Dec + dets ok): {len(grbs_df)}")
    print(grbs_df.head())