import os, shutil, json, sys, inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgb
import pandas as pd
import h5py
import arviz as az
from tqdm.auto import tqdm
import glob


from heapy.util.time import fermi_utc_to_met
from heapy.data.retrieve import gbmRetrieve
from heapy.geos.geometry import gbmGeometry
from heapy.pipe.event import gbmTTE
from bayspec.model.local import *
from bayspec import DataUnit, Data, BayesInfer, Plot, MaxLikeFit

# ============================================================================
# CHANGELOG — fixes applied during pipeline review (see chat history for derivations)
# ============================================================================
# - get_bayspec_path(grb_name): added. load_grb()'s `bayspec_data` global gets
#   overwritten by the last GRB processed in any loop — use this function for
#   reliable per-GRB HDF5 path access after a loop completes, instead of the
#   stale `bayspec_data` variable.
# - beta (band model): no longer hardcode one value for all GRBs — pass
#   fixed_params={'beta': <per-GRB value>} at both fit_all_time_slices() and
#   extract_params() call sites, using each GRB's own time-integrated band fit.
# - clean_dir(path, prefix=None): replaces old setup_slice_dirs(). Generic
#   reusable cleanup; slice dirs don't need pre-creation since downstream
#   os.makedirs(..., exist_ok=True) calls create them on demand.
# - compute_vFv(alpha, Ep, A, model_name): now takes model_name. cpl's
#   pivot_energy=1, band's pivot_energy=100 (bayspec defaults, confirmed from
#   source) — formula needs an epiv**(-alpha) factor that only cancels for cpl.
# - extract_tintegrated_spectra(): decoupled from time_slices. Needs its own
#   't1'/'t2' columns in grbs_df (pushed via load_grb as globals) — independent
#   window from the time-resolved slicing.
# - fit_tintegrated()/fit_all_time_slices(): rebn max_bin unified to 20
#   (was 10 vs 20, inconsistent). Both print a WARNING before overwriting an
#   existing fit directory.
# - get_model_params(): raises a clear ValueError on an empty posterior sample
#   (0 rows) instead of a bare IndexError on df_1sigma.iloc[0].
# - _load_and_fit() (inside extract_params): wrapped in try/except ValueError —
#   catches post_equal_weights.dat column-count mismatch vs fixed_params (and
#   the above empty-sample case) — returns None for that slice/GRB instead of
#   killing the whole extraction loop.
# - inspect/sys imports moved to module top. Were previously imported inside
#   load_grb(), so they leaked into the notebook's globals via the
#   locals()-based push (to_push) every time load_grb() ran.
# - extract_tintegrated_spectra()/extract_tresolved_spectra(): added a WIDER
#   filter_time() re-filter right before spec_slices/extract_spectrum/
#   extract_response (original narrow window kept, commented, for fallback).
#   Confirmed via heapy source: spec_t1t2 (derived from the filter_time range)
#   sets the bin range for the background polynomial fit — a narrow window
#   starves that fit of off-burst baseline.
# - bs_ignore (extract_tresolved_spectra only): scoped to the FULL burst span
#   (all slices), not just the current slice. Confirmed via heapy source
#   (pgSignal(..., ignore=self.bs_ignore) feeds the .bkg background fit
#   directly) — per-slice scoping let other slices' genuine burst emission
#   contaminate this slice's background estimate.
# - extract_rebin_curve() calls: commented out (unused), not deleted.
# - load_grb(): now creates bayspec_tresolved_path up front. bayspec_data (the
#   shared per-GRB h5 file) always lives there, even for tintegrated results —
#   without this, extract_params(mode='tintegrated') on a brand-new GRB crashed
#   with FileNotFoundError if the tresolved stage hadn't run yet for that GRB.
# ============================================================================

#------------------------------------------------------------------------------------------

def compute_vFv(alpha, Ep, A, model_name):
    """nuFnu at peak. epiv = pivot_energy, bayspec default per model:
    cpl: epiv=1, band: epiv=100. General form, same equation for both,
    only epiv changes."""
    epiv = {'cpl': 1.0, 'band': 100.0}[model_name]
    return 1.602e-9 * A * (Ep**(alpha + 2)) * epiv**(-alpha) * np.exp(-(2 + alpha))

#------------------------------------------------------------------------------------------
def view_hdf5(filepath, key=None):
    """View HDF5 file keys and optionally display a specific key's data."""
    with h5py.File(filepath, 'r') as f:
        available_keys = list(f.keys())
        print(f"Available keys: {available_keys}\n")
    
    if key and key in available_keys:
        return pd.read_hdf(filepath, key=key)
    elif key:
        print(f"ERROR: Key '{key}' not found")
#------------------------------------------------------------------------------------------

def show_directory_tree(path, max_depth=3, max_files=5):
    import os
    import subprocess
    
    print(f"Current working directory: {os.getcwd()}\n")
    print(f"=== Directory Tree: {path} ===\n")
    
    # Try using tree command first (cleaner output)
    try:
        result = subprocess.run(
            ['tree', '-L', str(max_depth), path], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(result.stdout)
            return
    except:
        pass
    
    # Fallback: custom tree display
    def print_tree(current_path, prefix="", depth=0):
        if depth >= max_depth:
            return
        
        try:
            items = sorted(os.listdir(current_path))
            dirs = [i for i in items if os.path.isdir(os.path.join(current_path, i))]
            files = [i for i in items if os.path.isfile(os.path.join(current_path, i))]
            
            # Print directories
            for i, d in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                connector = "└── " if is_last_dir else "├── "
                print(f"{prefix}{connector}{d}/")
                
                new_prefix = prefix + ("    " if is_last_dir else "│   ")
                print_tree(os.path.join(current_path, d), new_prefix, depth + 1)
            
            # Print files (limited)
            files_to_show = files[:max_files]
            remaining = len(files) - max_files
            
            for i, f in enumerate(files_to_show):
                is_last = (i == len(files_to_show) - 1) and remaining <= 0
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}{f}")
            
            if remaining > 0:
                print(f"{prefix}└── ... ({remaining} more files)")
                
        except PermissionError:
            print(f"{prefix}[Permission Denied]")
    
    print(f"{os.path.basename(path) or path}/")
    print_tree(path)
    print()


#------------------------------------------------------------------------------------------

def clean_dir(path, prefix=None):
    """Remove items under `path`. If `prefix` given, only items whose name
    starts with it are removed (dirs via rmtree, files via remove). Slice
    dirs don't need pre-creation — downstream os.makedirs(..., exist_ok=True)
    calls create them on demand."""
    if not os.path.exists(path):
        return
    for item in os.listdir(path):
        if prefix and not item.startswith(prefix):
            continue
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)
    print(f"Cleaned: {path}" + (f" (prefix='{prefix}')" if prefix else ""))

#------------------------------------------------------------------------------------------

# Base data directory (mounted from Mac)
DATA_BASE = '/workspace/data'

def get_bayspec_path(grb_name):
    """Reconstruct a GRB's bayspec_data.h5 path without needing load_grb()
    state (which gets overwritten on the last GRB processed in a loop)."""
    base = os.path.join(DATA_BASE, grb_name)
    return os.path.join(base, 'data/tresolved/bayspec', f'{grb_name}_bayspec_data.h5')

def load_grb(grb):
    # global GRB_Name, RA, DEC, utc, sel_dets, time_slices, n_slices
    # global tintegrated_path, heapy_tintegrated_path, bayspec_tintegrated_path
    # global tresolved_path, heapy_tresolved_path, bayspec_tresolved_path
    # global bayspec_data, Sampler_Output_DIR, Posterior_Stat_File
    # global fermi_met, gbm_rtv

    GRB_Name    = grb['name']
    RA          = grb['ra']
    DEC         = grb['dec']
    utc         = grb['utc']
    sel_dets    = grb['sel_dets']    if 'sel_dets'    in grb.index else None
    time_slices = grb['time_slices'] if 'time_slices' in grb.index else None
    n_slices    = len(time_slices)   if time_slices is not None    else None
    t1          = grb['t1']          if 't1'          in grb.index else None
    t2          = grb['t2']          if 't2'          in grb.index else None

    # Base paths (relative to DATA_BASE)
    base = os.path.join(DATA_BASE, GRB_Name)
    gbm_data = os.path.join(DATA_BASE, 'gbm_data')

    # Time-integrated analysis
    tintegrated_path         = os.path.join(base, 'data/tintegrated')
    heapy_tintegrated_path   = os.path.join(base, 'data/tintegrated/heapy')
    bayspec_tintegrated_path = os.path.join(base, 'data/tintegrated/bayspec')

    # Time-resolved analysis
    tresolved_path           = os.path.join(base, 'data/tresolved')
    heapy_tresolved_path     = os.path.join(base, 'data/tresolved/heapy')
    bayspec_tresolved_path   = os.path.join(base, 'data/tresolved/bayspec')

    # bayspec_data (below) always lives under bayspec_tresolved_path, even for
    # tintegrated results (single shared h5 file, different key prefixes) — so
    # this dir must exist before the FIRST extract_params() call, even if only
    # the tintegrated stage has run so far for this GRB.
    os.makedirs(bayspec_tresolved_path, exist_ok=True)

    # Output and Data Files
    bayspec_data             = os.path.join(bayspec_tresolved_path, f'{GRB_Name}_bayspec_data.h5')
    Sampler_Output_DIR       = os.path.join(base, 'data/Sampler_Output')
    Posterior_Stat_File      = os.path.join(Sampler_Output_DIR, f'{GRB_Name}_posterior_stat.h5')

    # Fermi
    fermi_met = fermi_utc_to_met(utc)
    gbm_rtv   = gbmRetrieve.from_utc(utc=utc, t1=-400, t2=400, datapath=gbm_data)

    print(f"Loaded: {GRB_Name} | RA={RA} | DEC={DEC} | {n_slices} slices | tint_window=({t1},{t2}) | dets={sel_dets}")
    to_push = {k: v for k, v in locals().items() if k != 'grb'}
    
    # Push to caller's notebook globals
    caller_globals = inspect.stack()[1].frame.f_globals
    caller_globals.update(to_push)
    
    # Push to grb_utils module's own globals (so other util functions see them too)
    sys.modules[__name__].__dict__.update(to_push)
#------------------------------------------------------------------------------------------
def extract_tintegrated_spectra():
    """Extract time-integrated spectra for all detectors.
    Single window [t1, t2] (seconds rel. to trigger) — independent of
    time_slices. Set 't1'/'t2' columns in grbs_df; pushed via load_grb()."""
    if t1 is None or t2 is None:
        raise ValueError(f"{GRB_Name}: t1/t2 not set — add 't1','t2' columns to grbs_df")

    spec_name = 'tintegrated'
    
    spec_dir = heapy_tintegrated_path
    os.makedirs(spec_dir, exist_ok=True)
    
    # Clean old files
    for f in os.listdir(spec_dir):
        fpath = os.path.join(spec_dir, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    
    print(f"\nProcessing time-integrated spectra for {GRB_Name}")
    
    for det in sel_dets:
        print(f"Processing {det}...")
        try:
            gbm_tte = gbmTTE(gbm_rtv.rtv_res['tte'][det], gbm_rtv.rtv_res['poshist'][0])
            
            time_offset = fermi_met - gbm_tte.timezero
            print(f"Filtering around GRB with offset {time_offset:.2f}s")
            
            gbm_tte.event
            gbm_tte.filter_time([time_offset - 20, time_offset + 50])
            
            if det[0] == 'n':
                gbm_tte.filter_energy([8, 900])
            else:
                gbm_tte.filter_energy([300, 38000])
            
            temp_dir = f'{spec_dir}/temp_{det}'
            os.makedirs(temp_dir, exist_ok=True)
            
            gbm_tte.lc_binsize = 0.5
            gbm_tte.extract_curve(savepath=temp_dir, show=False)
            # gbm_tte.extract_rebin_curve(min_sigma=2, savepath=temp_dir, loglog=False, step=True, show=False)  # unused for now, commented per request
            gbm_tte.calculate_txx(xx=0.9, savepath=temp_dir)
            
            # Wider re-filter before spectral extraction — heapy docs pattern: spectrum/
            # response need a wider pre/post-burst baseline than the lightcurve for the
            # background polynomial fit. Matches load_grb's gbmRetrieve window (t1=-400, t2=400).
            # gbm_tte.filter_time([time_offset - 20, time_offset + 50])  # original narrow window, kept for fallback
            gbm_tte.filter_time([time_offset - 380, time_offset + 380])
            gbm_tte.spec_slices = [[time_offset + t1, time_offset + t2]]
            gbm_tte.extract_spectrum(savepath=temp_dir, show=False)
            gbm_tte.extract_response(ra=RA, dec=DEC, savepath=temp_dir)
            
            # Rename files
            saved_name = None
            for f in os.listdir(temp_dir):
                if f.endswith('.src'):
                    saved_name = f.replace('.src', '')
                    break
            
            if saved_name:
                for ext in ['src', 'bkg', 'rsp']:
                    src_file = f'{temp_dir}/{saved_name}.{ext}'
                    dst_file = f'{spec_dir}/{spec_name}_{det}.{ext}'
                    if os.path.exists(src_file):
                        shutil.move(src_file, dst_file)
            
            shutil.rmtree(temp_dir)
            print(f"Saved {det} to {spec_dir}")
            
        except Exception as e:
            print(f"Error processing {det}: {e}")
            continue
    
    print(f"Time-integrated spectra saved to: {spec_dir}")

#------------------------------------------------------------------------------------------
def fit_tintegrated(model_name='band', fixed_params=None, nlive=1000, skip_dets=None):
    """Fit time-integrated spectrum."""
    if fixed_params is None:
        fixed_params = {}
    if skip_dets is None:
        skip_dets = []
    
    print(f"\nFitting {model_name} (tint) | {GRB_Name}")
    
    savepath = f'{bayspec_tintegrated_path}/{model_name}'
    os.makedirs(savepath, exist_ok=True)
    
    if os.listdir(savepath):
        print(f"  WARNING: overwriting existing fit at {savepath}")
    
    for item in os.listdir(savepath):
        p = os.path.join(savepath, item)
        os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)
    
    data_list = []
    for det in sel_dets:
        if det in skip_dets:
            continue
        try:
            base = os.path.join(heapy_tintegrated_path, f'tintegrated_{det}')
            notc = [8, 900] if det[0] == 'n' else [300, 38000]
            du = DataUnit(src=f'{base}.src', bkg=f'{base}.bkg', rsp=f'{base}.rsp',
                         notc=notc, stat='pgstat', rebn={'min_sigma': 2, 'max_bin': 20})
            data_list.append((det, du))
        except Exception as e:
            print(f"  Skip {det}: {e}")
    
    if len(data_list) < 2:
        print("ERROR: <2 detectors")
        return
    
    data = Data(data_list)
    data.save(savepath)
    
    model = {'cpl': cpl, 'band': band, 'pl': pl}.get(model_name, None)()
    for pname, pval in fixed_params.items():
        if hasattr(model, pname):
            setattr(model, pname, pval)
    model.save(savepath)
    
    infer = BayesInfer([(data, model)])
    infer.save(savepath)
    post = infer.multinest(nlive=nlive, resume=False, verbose=False, savepath=savepath)
    post.save(savepath)
    
    # Plots — write directly, skip PDF
    Plot.infer(post, style='CE', ploter='matplotlib').fig.savefig(f'{savepath}/ctsspec.png', dpi=100, bbox_inches='tight')
    Plot.infer(post, style='NE', ploter='matplotlib').fig.savefig(f'{savepath}/phtspec.png', dpi=100, bbox_inches='tight')

    modelplot = Plot.model(ploter='matplotlib', style='vFv', post=True)
    modelplot.add_model(model, E=np.logspace(1, 3, 100))
    modelplot.fig.savefig(f'{savepath}/model.png', dpi=100, bbox_inches='tight')

    print(f"✓ {model_name}")
#------------------------------------------------------------------------------------------
def extract_params(model_name='band', mode='tresolved', fixed_params=None):
    """Extract spectral parameters (best-fit + 1-sigma range) for either
    time-resolved (per slice) or time-integrated spectra.
    
    Args:
        model_name: 'cpl', 'band', 'pl'
        mode: 'tresolved' or 'tintegrated'
        fixed_params: dict of frozen params, e.g. {'beta': -5.51154}
    """
    if fixed_params is None:
        fixed_params = {}
    
    model_cols = {
        'cpl':  ['alpha', 'log_Ep', 'log_A'],
        'band': ['alpha', 'beta', 'log_Ep', 'log_A'],
        'pl':   ['alpha', 'log_A'],
    }
    if model_name not in model_cols:
        raise ValueError(f"Unknown model: {model_name}")
    
    free_params = [p for p in model_cols[model_name] if p not in fixed_params]
    columns = free_params + ['log_likelihood']
    
    def _load_and_fit(savepath):
        fpath = f'{savepath}/1-post_equal_weights.dat'
        if not os.path.exists(fpath):
            return None
        
        try:
            samples = np.loadtxt(fpath)
            df = pd.DataFrame(samples, columns=columns)
            
            # Add frozen params as constant columns (so get_model_params works unchanged)
            for pname, pval in fixed_params.items():
                df[pname] = pval
            
            df_sorted = df.sort_values('log_likelihood', ascending=False).reset_index(drop=True)
            
            if model_name != 'pl':
                df_sorted['Ep'] = 10**df_sorted['log_Ep']
            df_sorted['A'] = 10**df_sorted['log_A']
            if model_name != 'pl':
                df_sorted['vFv'] = compute_vFv(df_sorted['alpha'], df_sorted['Ep'], df_sorted['A'], model_name)
            
            n_total = len(df_sorted)
            n_1sigma = max(int(0.6827 * n_total), 1)
            df_1sigma = df_sorted.iloc[:n_1sigma].copy()
            
            return get_model_params(model_name, df_1sigma)
        except ValueError as e:
            print(f"  ValueError at {fpath}: {e}")
            print(f"  (check post_equal_weights.dat column count matches fixed_params={fixed_params} used at fit time)")
            return None
    
    if mode == 'tintegrated':
        print(f"\nExtracting {model_name} (tint) | {GRB_Name}")
        savepath = f'{bayspec_tintegrated_path}/{model_name}'
        row = _load_and_fit(savepath)
        if row is None:
            print("  File not found")
            return None
        
        df_out = pd.DataFrame([row])
        df_out.to_hdf(bayspec_data, key=f'tint_{model_name}_{GRB_Name}', mode='a')
        print(f"  ✓ Saved tint_{model_name}_{GRB_Name}")
        return df_out
    
    elif mode == 'tresolved':
        print(f"\nExtracting {model_name} (tresolved) | {GRB_Name}")
        rows = []
        for slice_idx in range(1, n_slices + 1):
            savepath = f'{bayspec_tresolved_path}/{model_name}/slice_{slice_idx:02d}'
            row = _load_and_fit(savepath)
            if row is None:
                print(f"  Skipping slice {slice_idx}: file not found")
                continue
            
            t_start, t_stop = time_slices[slice_idx - 1]
            rows.append({
                'slice': slice_idx,
                'model': model_name,
                't_start': t_start,
                't_stop': t_stop,
                **row,
            })
        
        df_out = pd.DataFrame(rows).sort_values('slice').reset_index(drop=True)
        df_out.to_hdf(bayspec_data, key=model_name, mode='a')
        print(f"  ✓ Saved {model_name}")
        return df_out
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def get_model_params(model_name, df_1sigma):
    """Extract best-fit and range parameters for a given model.
    
    Args:
        model_name: 'cpl', 'band', or 'pl'
        df_1sigma: DataFrame with 1-sigma samples (sorted by likelihood)
    
    Returns:
        row: dict with best-fit and uncertainty values
    """
    if len(df_1sigma) == 0:
        raise ValueError(
            f"get_model_params('{model_name}'): empty 1-sigma sample (0 rows) — "
            f"check the fit converged and post_equal_weights.dat has data"
        )
    
    # Common extractions
    row = {}
    
    # Alpha (all models)
    alpha_best = df_1sigma.iloc[0]['alpha']
    alpha_min, alpha_max = np.min(df_1sigma['alpha']), np.max(df_1sigma['alpha'])
    row.update({
        'alpha': alpha_best,
        'alpha_low': alpha_best - alpha_min,
        'alpha_high': alpha_max - alpha_best,
    })

    # Beta (band only)
    if model_name == 'band':
        beta_best = df_1sigma.iloc[0]['beta']
        beta_min, beta_max = np.min(df_1sigma['beta']), np.max(df_1sigma['beta'])
        row.update({
            'beta': beta_best,
            'beta_low': beta_best - beta_min,
            'beta_high': beta_max - beta_best,
        })
    
    # A (amplitude, all models)
    A_best = df_1sigma.iloc[0]['A']
    row['A'] = A_best

    # Ep & vFv (all models except 'pl')
    if model_name != 'pl':
        Ep_best = df_1sigma.iloc[0]['Ep']
        vFv_best = df_1sigma.iloc[0]['vFv']
        Ep_min, Ep_max = np.min(df_1sigma['Ep']), np.max(df_1sigma['Ep'])
        vFv_min, vFv_max = np.min(df_1sigma['vFv']), np.max(df_1sigma['vFv'])
        
        row.update({
            'Ep_best': Ep_best,
            'Ep_low': Ep_best - Ep_min,
            'Ep_high': Ep_max - Ep_best,
            'vFv_best': vFv_best,
            'vFv_low': vFv_best - vFv_min,
            'vFv_high': vFv_max - vFv_best,
            'sigma_Ep': (Ep_max - Ep_min) / 2,
            'sigma_vFv': (vFv_max - vFv_min) / 2,
        })
    
    return row
#------------------------------------------------------------------------------------------
def extract_tresolved_spectra():
    """Extract time-resolved spectra for current GRB (all slices, all detectors)."""
    
    print(f"#-----------------------x| for {GRB_Name} |x------------------------#")
    
    clean_dir(heapy_tresolved_path, prefix='slice_')
    
    for slice_index, (t_start, t_stop) in enumerate(time_slices, 1):
        if t_start < 0:
            spec_name = f'm{abs(t_start):.2f}'.replace('.', 'd') + f'_p{t_stop:.2f}'.replace('.', 'd')
        else:
            spec_name = f'p{t_start:.2f}'.replace('.', 'd') + f'_p{t_stop:.2f}'.replace('.', 'd')

        print(f"\nProcessing slice {slice_index}: {spec_name} ({t_start}, {t_stop})")

        slice_dir = f'{heapy_tresolved_path}/slice_{slice_index:02d}'

        for det in sel_dets:
            print(f"Processing {det}...")
            try:
                gbm_tte = gbmTTE(gbm_rtv.rtv_res['tte'][det], gbm_rtv.rtv_res['poshist'][0])

                time_offset = fermi_met - gbm_tte.timezero
                print(f"Filtering around GRB with offset {time_offset:.2f}s")

                gbm_tte.event
                gbm_tte.filter_time([time_offset - 20, time_offset + 50])

                if det[0] == 'n':
                    gbm_tte.filter_energy([8, 900])
                else:
                    gbm_tte.filter_energy([300, 38000])

                temp_dir = f'{slice_dir}/temp_{det}'
                os.makedirs(temp_dir, exist_ok=True)

                gbm_tte.lc_binsize = 0.5
                # bs_ignore masks the burst emission from the background polynomial fit.
                # Scoped to the FULL burst span (all slices), not just this slice's window —
                # otherwise other slices (still genuine burst emission, not background) would
                # leak into this slice's background fit and bias the result.
                gbm_tte.bs_ignore = [[time_offset + time_slices[0][0], time_offset + time_slices[-1][1]]]
                gbm_tte.bs_deg = 1

                gbm_tte.extract_curve(savepath=temp_dir, show=False)
                # gbm_tte.extract_rebin_curve(min_sigma=2, savepath=temp_dir, loglog=False, step=True, show=False)  # unused for now, commented per request
                gbm_tte.calculate_txx(xx=0.9, savepath=temp_dir)

                # Wider re-filter before spectral extraction — heapy docs pattern: spectrum/
                # response need a wider pre/post-burst baseline than the lightcurve for the
                # background polynomial fit. Matches load_grb's gbmRetrieve window (t1=-400, t2=400).
                # gbm_tte.filter_time([time_offset - 20, time_offset + 50])  # original narrow window, kept for fallback
                gbm_tte.filter_time([time_offset - 380, time_offset + 380])
                gbm_tte.spec_slices = [[time_offset + t_start, time_offset + t_stop]]
                gbm_tte.extract_spectrum(savepath=temp_dir, show=False)
                gbm_tte.extract_response(ra=RA, dec=DEC, savepath=temp_dir)

                # Find actual filename heapy saved, rename to spec_name convention
                saved_name = None
                for f in os.listdir(temp_dir):
                    if f.endswith('.src'):
                        saved_name = f.replace('.src', '')
                        break

                if saved_name:
                    for ext in ['src', 'bkg', 'rsp']:
                        src_file = f'{temp_dir}/{saved_name}.{ext}'
                        dst_file = f'{slice_dir}/{spec_name}_{det}.{ext}'
                        if os.path.exists(src_file):
                            shutil.move(src_file, dst_file)
                        else:
                            print(f"  Warning: {ext} not found: {src_file}")
                else:
                    print(f"  Warning: no .src file found in {temp_dir}")

                shutil.rmtree(temp_dir)
                print(f"Saved {det} to {slice_dir}")

            except Exception as e:
                print(f"Error processing {det}: {e}")
                continue

        print(f"Completed slice {slice_index}")

    print("\nAll time-sliced spectra extracted and organized!")
    print(f"Files saved in: {heapy_tresolved_path}")
    

def fit_all_time_slices(time_slices, model_name='cpl', fixed_params=None):
    
    """Fit all time slices.
    
    Args:
        time_slices: list of (t_start, t_stop) tuples
        model_name: 'cpl', 'band', 'pl'
        fixed_params: dict like {'beta': -2.3} to fix parameters
    """
    if fixed_params is None:
        fixed_params = {}
    
    # Map plain names to bayspec's LaTeX param keys
    param_key_map = {
        'alpha': '$\\alpha$',
        'beta': '$\\beta$',
        'Ep': 'log$E_p$',
        'log_Ep': 'log$E_p$',
        'A': 'log$A$',
        'log_A': 'log$A$',
    }
    
    model_dir = f'{bayspec_tresolved_path}/{model_name}'
    if os.path.isdir(model_dir):
        existing_slices = [i for i in os.listdir(model_dir) if i.startswith('slice_')]
        if existing_slices:
            print(f"  WARNING: overwriting {len(existing_slices)} existing slice fit(s) in {model_dir}")
        for item in existing_slices:
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Cleaned: {item_path}")
    
    for slice_index, (t_start, t_stop) in enumerate(time_slices, 1):
        if t_start < 0:
            spec_name = f'm{abs(t_start):.2f}'.replace('.', 'd') + f'_p{t_stop:.2f}'.replace('.', 'd')
        else:
            spec_name = f'p{t_start:.2f}'.replace('.', 'd') + f'_p{t_stop:.2f}'.replace('.', 'd')
        
        print(f"\n{'='*50}")
        print(f"Processing slice {slice_index}: [{t_start}, {t_stop}] seconds")
        if fixed_params:
            print(f"Fixed: {fixed_params}")
        print(f"{'='*50}")
        
        savepath = f'{bayspec_tresolved_path}/{model_name}/slice_{slice_index:02d}'
        os.makedirs(savepath, exist_ok=True)

        if os.listdir(savepath):
            print(f"  WARNING: overwriting existing fit at {savepath}")

        for item in os.listdir(savepath):
            item_path = os.path.join(savepath, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        try:
            slice_dir = f'{heapy_tresolved_path}/slice_{slice_index:02d}'

            def spec_files(det):
                base = os.path.join(slice_dir, f'{spec_name}_{det}')
                return dict(src=f'{base}.src', bkg=f'{base}.bkg', rsp=f'{base}.rsp')

            data_list = []
            for det in sel_dets:
                try:
                    notc = [8, 900] if det.startswith('n') else [300, 38000]
                    du = DataUnit(**spec_files(det), notc=notc, stat='pgstat',
                                 rebn={'min_sigma': 2, 'max_bin': 20})
                    data_list.append((det, du))
                except Exception as e:
                    print(f"Skipping detector {det} in slice {slice_index}: {e}")
                    continue

            if len(data_list) < 2:
                print(f"Skipping slice {slice_index}, not enough valid detectors")
                continue

            data = Data(data_list)
            data.save(savepath)
            
            model = {'cpl': cpl, 'band': band, 'pl': pl}.get(model_name, None)()
            
            # Freeze parameters using correct API
            for pname, pval in fixed_params.items():
                key = param_key_map.get(pname, pname)
                if key in model.params:
                    model.params[key].frozen_at(pval)
                    print(f"  Frozen {pname} = {pval}")
                else:
                    print(f"  WARNING: param '{pname}' (key '{key}') not found in model")
            
            model.save(savepath)
            
            infer = BayesInfer([(data, model)])
            infer.save(savepath)
            
            post = infer.multinest(nlive=1000, resume=False, verbose=False, savepath=savepath)
            post.save(savepath)

            Plot.infer(post, style='CE', ploter='matplotlib').fig.savefig(f'{savepath}/ctsspec.png', dpi=100, bbox_inches='tight')
            Plot.infer(post, style='NE', ploter='matplotlib').fig.savefig(f'{savepath}/phtspec.png', dpi=100, bbox_inches='tight')
            
            modelplot = Plot.model(ploter='matplotlib', style='vFv', post=True)
            modelplot.add_model(model, E=np.logspace(1, 3, 100))
            modelplot.fig.savefig(f'{savepath}/model.png', dpi=100, bbox_inches='tight')
            
            print(f"Successfully processed slice {slice_index}")
            
        except Exception as e:
            print(f"Error processing slice {slice_index}: {e}")
            continue
        
    print("\nAll slices fitted!")

#------------------------------------------------------------------------------------------
# Plotting Fuctions
#------------------------------------------------------------------------------------------
# def plot_vFv_Ep(model='cpl', xlim=(1e0, 1e4), ylim=(1e-9, 1e-5), cmap='Reds_r', figsize=(8, 6)):
#     """Plot vFv vs Ep evolution for a single model.
    
#     Args:
#         model: str ('cpl', 'band', 'pl')
#         xlim: tuple (xmin, xmax)
#         ylim: tuple (ymin, ymax)
#         cmap: colormap name
#         figsize: tuple (width, height)
#     """
#     fig, ax = plt.subplots(figsize=figsize)
#     plt.style.use("seaborn-v0_8-whitegrid")
    
#     try:
#         SED_Par = pd.read_hdf(bayspec_data, key=model)
#     except:
#         print(f"Key not found: vFv_Ep_{model}_{GRB_Name}")
#         return
    
#     SED_Par['t_mid'] = (SED_Par['t_start'] + SED_Par['t_stop']) / 2

#     norm = Normalize(vmin=SED_Par['t_mid'].min(), vmax=SED_Par['t_mid'].max())
    
#     # Reference lines
#     Ep_line = np.logspace(1, 4, 200)
#     k_base = 1e-12
#     for factor in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
#         ax.plot(Ep_line, factor * k_base * Ep_line**3, ls='dashed', lw=0.8, color='gray', alpha=0.3)
    
#     # Error bars
#     ax.errorbar(SED_Par["Ep_best"], SED_Par["vFv_best"],
#                xerr=[SED_Par["Ep_low"], SED_Par["Ep_high"]],
#                yerr=[SED_Par["vFv_low"], SED_Par["vFv_high"]],
#                fmt='none', ecolor='black', alpha=0.5, linewidth=1, zorder=2, capsize=3)
    
#     # Scatter plot
#     sc = ax.scatter(SED_Par["Ep_best"], SED_Par["vFv_best"], c=SED_Par["t_mid"],
#                    cmap=cmap, s=80, norm=norm, linewidths=0.8, zorder=3, edgecolors='black')
    
#     # Start/End markers
#     ax.scatter(SED_Par["Ep_best"].iloc[0], SED_Par["vFv_best"].iloc[0],
#               marker="X", s=150, color="#E60000", edgecolors='black', linewidths=0.8, label="Start", zorder=4)
#     ax.scatter(SED_Par["Ep_best"].iloc[-1], SED_Par["vFv_best"].iloc[-1],
#               marker="X", s=120, color="#FFB0B0", edgecolors='black', linewidths=0.8, label="End", zorder=4)
    
#     # Formatting
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.set_xlabel(r"$E_p$ (keV)", fontsize=12)
#     ax.set_ylabel(r"$\nu F_\nu(E_p)$ [erg cm$^{-2}$ s$^{-1}$]", fontsize=12)
#     ax.set_title(f"{GRB_Name}: $E_p$–Flux Evolution ({model.upper()})", fontsize=13, pad=10)
#     ax.legend(frameon=True, fontsize=10)
#     ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
    
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label("Time (s)")
    
#     plt.tight_layout()
#     savename = f'{bayspec_tresolved_path}/{GRB_Name}_vFv_Ep_{model}.png'
#     plt.savefig(savename, dpi=300)
#     print(f"Saved: {savename}")
#     plt.show()

#------------------------------------------------------------------------------------------
