"""
grb_pipeline_extras.py
-----------------------
Steps 2-4 of the pipeline. Builds on your existing grb_utils.py (load_grb,
extract_tintegrated_spectra, etc.) — nothing in grb_utils.py is modified,
this only adds new functions and imports grb_utils' current globals via the
same module-global push pattern load_grb() already uses.

Step 3 note (read before running):
    extract_lc_array() needs the raw TTE photon time array off the gbm_tte
    object. The first guess (gbm_tte.data.time) was wrong — confirmed by a
    real run: 'gbmTTE' object has no attribute 'data'. It now tries several
    plausible attribute paths (_get_event_times()) in order; if all of them
    fail it raises an error listing dir(gbm_tte) so the correct attribute
    name is visible directly in the traceback on the next run — at that
    point add the right one as the first entry in _get_event_times().

Install (beyond what heapy/bayspec already need):
    pip install astroquery --break-system-packages
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import grb_utils as gu
from grb_utils import load_grb, extract_tintegrated_spectra, fit_tintegrated
from heapy.pipe.event import gbmTTE


# ----------------------------------------------------------------------------
# Step 3: light-curve + spectrum array extraction ("both, so we can eyeball
# which bursts have a FRED shape").
# ----------------------------------------------------------------------------

def _get_event_times(gbm_tte):
    """Return the raw TTE photon arrival-time array off a gbm_tte object.

    We don't have confirmed source access to this heapy version's gbmTTE
    class, so rather than hard-coding one guessed attribute path (which
    failed: 'gbmTTE' object has no attribute 'data'), this tries several
    plausible ones in order. If ALL of them fail, it raises with the full
    dir(gbm_tte) listing attached — so the traceback itself tells us the
    real attribute name in one shot, instead of another guess-and-check
    round trip.
    """
    candidates = [
        lambda t: t.event.time,
        lambda t: t.event['TIME'],
        lambda t: t.time,
        lambda t: t.times,
        lambda t: t.evt.time,
        lambda t: t.data['TIME'],
        lambda t: t.event.TIME,
    ]
    errors = []
    for get in candidates:
        try:
            return np.asarray(get(gbm_tte))
        except Exception as e:
            errors.append(f"{get}: {type(e).__name__}: {e}")
            continue

    available = [a for a in dir(gbm_tte) if not a.startswith('_')]
    raise AttributeError(
        "Could not find TTE event times on gbm_tte via any known attribute "
        f"path. Attempts tried:\n  " + "\n  ".join(errors) +
        f"\n\nAvailable (non-private) attributes on gbm_tte:\n  {available}\n"
        "Pick the right one from this list and tell me — I'll wire it in as "
        "the first entry in _get_event_times()'s candidates list."
    )


def extract_lc_array(det, binsize=0.064, pre=-20, post=50, savepath=None,
                      plot=True):
    """Bin raw TTE photon arrival times into a light curve array and save it
    as {GRB_Name}_{det}_lc.npy (shape (N,2): [t_center, rate]) + a quicklook
    PNG. Uses the CURRENT globals pushed by grb_utils.load_grb() (GRB_Name,
    fermi_met, gbm_rtv), so call load_grb(row) before this.

    This is the array to look at for FRED-shape screening — it's the time
    profile.
    """
    savepath = savepath or gu.heapy_tintegrated_path
    os.makedirs(savepath, exist_ok=True)

    gbm_tte = gbmTTE(gu.gbm_rtv.rtv_res['tte'][det], gu.gbm_rtv.rtv_res['poshist'][0])
    time_offset = gu.fermi_met - gbm_tte.timezero

    gbm_tte.event  # trigger lazy load, same as existing grb_utils.py calls
    gbm_tte.filter_time([time_offset + pre, time_offset + post])
    if det[0] == 'n':
        gbm_tte.filter_energy([8, 900])
    else:
        gbm_tte.filter_energy([300, 38000])

    times = _get_event_times(gbm_tte)

    t_rel = times - gu.fermi_met  # seconds relative to trigger
    bins = np.arange(pre, post + binsize, binsize)
    counts, edges = np.histogram(t_rel, bins=bins)
    t_centers = 0.5 * (edges[:-1] + edges[1:])
    rate = counts / binsize

    arr = np.column_stack([t_centers, rate])
    npy_path = os.path.join(savepath, f'{gu.GRB_Name}_{det}_lc.npy')
    np.save(npy_path, arr)
    
    if not os.path.exists(npy_path):
        raise IOError(f"Failed to save {npy_path} — file not created after np.save()")

    if plot:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.step(t_centers, rate, where='mid', lw=0.8)
        ax.set_xlabel('Time since trigger (s)')
        ax.set_ylabel('Rate (cts/s)')
        ax.set_title(f'{gu.GRB_Name} — {det} light curve')
        fig.tight_layout()
        png_path = os.path.join(savepath, f'{gu.GRB_Name}_{det}_lc.png')
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        
        if not os.path.exists(png_path):
            raise IOError(f"Failed to save {png_path} — file not created after fig.savefig()")

    return arr


def extract_spectrum_array(det, src_path, savepath=None, plot=True):
    """Read a heapy-produced .src PHA file (already extracted by
    extract_tintegrated_spectra / extract_tresolved_spectra) into a plain
    (channel, counts) array and save as {GRB_Name}_{det}_spec.npy + PNG.

    This is the count-spectrum array — useful for spectral shape, not for
    FRED screening, but you asked for both to be available.
    """
    from astropy.io import fits

    savepath = savepath or os.path.dirname(src_path)
    os.makedirs(savepath, exist_ok=True)

    with fits.open(src_path) as hdul:
        data = hdul['SPECTRUM'].data if 'SPECTRUM' in hdul else hdul[1].data
        channels = np.asarray(data['CHANNEL'])
        counts = np.asarray(data['COUNTS'])

    arr = np.column_stack([channels, counts])
    npy_path = os.path.join(savepath, f'{gu.GRB_Name}_{det}_spec.npy')
    np.save(npy_path, arr)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.step(channels, counts, where='mid', lw=0.8)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        ax.set_title(f'{gu.GRB_Name} — {det} count spectrum')
        fig.tight_layout()
        fig.savefig(os.path.join(savepath, f'{gu.GRB_Name}_{det}_spec.png'), dpi=120)
        plt.close(fig)

    return arr


def extract_arrays_for_current_grb(binsize=0.064, pre=-20, post=50):
    """Run extract_lc_array() (+ extract_spectrum_array() if a matching .src
    already exists from extract_tintegrated_spectra()) for every detector of
    the currently loaded GRB. Call load_grb(row) first."""
    print(f"\n[ARRAYS] Starting array extraction...")
    print(f"[ARRAYS] gu.GRB_Name = {gu.GRB_Name}")
    print(f"[ARRAYS] gu.sel_dets = {gu.sel_dets}")
    print(f"[ARRAYS] gu.tintegrated_path = {gu.tintegrated_path}")
    
    savepath = os.path.join(gu.tintegrated_path, 'arrays')
    os.makedirs(savepath, exist_ok=True)
    print(f"[ARRAYS] Created/verified directory: {savepath}")

    for det in gu.sel_dets:
        print(f"  Attempting lc_array for {det}...")
        try:
            extract_lc_array(det, binsize=binsize, pre=pre, post=post, savepath=savepath)
            print(f"    ✓ lc_array {det} saved")
        except Exception as e:
            print(f"    ✗ lc array failed for {det}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        src_path = os.path.join(gu.heapy_tintegrated_path, f'tintegrated_{det}.src')
        if os.path.exists(src_path):
            print(f"  Attempting spectrum_array for {det}...")
            try:
                extract_spectrum_array(det, src_path, savepath=savepath)
                print(f"    ✓ spectrum_array {det} saved")
            except Exception as e:
                print(f"    ✗ spectrum array failed for {det}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Skipping spectrum_array for {det}: no .src file")


# ----------------------------------------------------------------------------
# Step 4: batch driver — discovery-fed df -> download -> tint extraction ->
# array extraction, one GRB at a time, failures isolated and logged.
# ----------------------------------------------------------------------------

def run_batch(grbs_df, do_tint_fit=False, tint_model='band', fixed_params=None,
              binsize=0.064, pre=-20, post=50, log_path='batch_errors.log'):
    """Loop over a grbs_df built by grb_discovery.build_grbs_df().

    For each GRB: load_grb -> extract_tintegrated_spectra -> (optional)
    fit_tintegrated -> extract arrays (lc + spectrum, both). Any failure at
    any stage is caught, logged to log_path, and the loop moves to the next
    GRB rather than dying partway through a catalog-sized batch.

    Returns (ok_names, failed_names).
    """
    ok, failed = [], []

    with open(log_path, 'a') as logf:
        for _, row in tqdm(grbs_df.iterrows(), total=len(grbs_df), desc='Batch GRBs'):
            name = row['name']
            try:
                load_grb(row)
                extract_tintegrated_spectra()

                if do_tint_fit:
                    fit_tintegrated(model_name=tint_model, fixed_params=fixed_params)

                extract_arrays_for_current_grb(binsize=binsize, pre=pre, post=post)

                ok.append(name)

            except Exception as e:
                msg = f"{name}: {type(e).__name__}: {e}"
                print(f"  FAILED — {msg}")
                logf.write(msg + '\n')
                failed.append(name)
                continue

    print(f"\nBatch done: {len(ok)} ok, {len(failed)} failed.")
    if failed:
        print(f"Failed GRBs (see {log_path}): {failed}")
    return ok, failed


def resume_batch(grbs_df, do_tint_fit=False, tint_model='band', fixed_params=None,
                 binsize=0.064, pre=-20, post=50, log_path='batch_errors.log',
                 skip_existing=True):
    """Resume a batch run, skipping GRBs that already have light-curve arrays.

    Args:
        grbs_df: full dataframe of GRBs to process
        skip_existing: if True, skip any GRB where {GRB_Name}_{det}_lc.npy 
                       already exists (i.e., arrays/ directory has files)
        (rest same as run_batch)

    Returns (ok_names, failed_names, skipped_names).
    """
    ok, failed, skipped = [], [], []

    with open(log_path, 'a') as logf:
        for _, row in tqdm(grbs_df.iterrows(), total=len(grbs_df), desc='Resume Batch'):
            name = row['name']
            
            # Check if already processed
            if skip_existing:
                arrays_dir = os.path.join(f'/workspace/data/{name}/data/tintegrated/arrays')
                if os.path.exists(arrays_dir):
                    lc_files = [f for f in os.listdir(arrays_dir) if f.endswith('_lc.npy')]
                    if lc_files:
                        print(f"  [SKIP] {name} — arrays already exist ({len(lc_files)} lc files)")
                        skipped.append(name)
                        continue
            
            try:
                load_grb(row)
                extract_tintegrated_spectra()

                if do_tint_fit:
                    fit_tintegrated(model_name=tint_model, fixed_params=fixed_params)

                extract_arrays_for_current_grb(binsize=binsize, pre=pre, post=post)

                ok.append(name)

            except Exception as e:
                msg = f"{name}: {type(e).__name__}: {e}"
                print(f"  FAILED — {msg}")
                logf.write(msg + '\n')
                logf.flush()
                failed.append(name)
                continue

    print(f"\nResume done: {len(ok)} newly processed, {len(failed)} failed, {len(skipped)} skipped (already done).")
    if failed:
        print(f"Failed GRBs (see {log_path}): {failed}")
    return ok, failed, skipped