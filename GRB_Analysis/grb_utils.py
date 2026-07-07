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

# Base data directory (mounted from Mac)
DATA_BASE = '/workspace/data'

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
def extract_geometry(dets=None):
    """SAA passage, visibility, detector angles, skymap — one-time per GRB."""
    if dets is None:
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']

    geo_dir = f'{base}/geometry'  # base is pushed by load_grb() too
    os.makedirs(geo_dir, exist_ok=True)

    gbm_geo = gbmGeometry(file=gbm_rtv.rtv_res['poshist'])
    print("saa_passage:", gbm_geo.saa_passage)
    print("location_visible:", gbm_geo.get_location_visible(
        ra=RA, dec=DEC, met=[fermi_met - 500, fermi_met, fermi_met + 500]))

    for det in dets:
        angle = gbm_geo.get_detector_angle(
            ra=RA, dec=DEC, det=det, met=[fermi_met - 100, fermi_met, fermi_met + 100])
        print(det, angle)

    gbm_geo.extract_skymap(ra=RA, dec=DEC, met=fermi_met, savepath=geo_dir)

#------------------------------------------------------------------------------------------
def extract_tintegrated_spectra():
    """Extract time-integrated lightcurve + spectrum, one det, teaching version."""
    for det in sel_dets:
        print(f"\n{det}")
        det_dir = f'{heapy_tintegrated_path}/{det}'
        os.makedirs(det_dir, exist_ok=True)

        gbm_tte = gbmTTE(gbm_rtv.rtv_res['tte'][det], gbm_rtv.rtv_res['poshist'][0])
        gbm_tte.event

        # lightcurve
        gbm_tte.filter_time([-200, 300], fermi_met)
        gbm_tte.filter_energy([8, 900] if det[0] == 'n' else [300, 38000])
        gbm_tte.lc_binsize = 0.5
        gbm_tte.extract_curve(savepath=det_dir, show=True)
        gbm_tte.calculate_txx(xx=0.9, savepath=det_dir)

        # spectrum — wider window needed for bkg fit
        gbm_tte.filter_time([-400, 400], fermi_met)
        gbm_tte.spec_slices = [[0, 70]]
        gbm_tte.extract_spectrum(savepath=det_dir, show=True)
        gbm_tte.extract_response(ra=RA, dec=DEC, savepath=det_dir)

        # flatten: heapy writes auto-named .src/.bkg/.rsp inside det_dir, but
        # fit_tintegrated() expects them flat as tintegrated_{det}.* directly
        # under heapy_tintegrated_path — find + copy them up with right name.
        saved_name = None
        for f in os.listdir(det_dir):
            if f.endswith('.src'):
                saved_name = f.replace('.src', '')
                break

        if saved_name:
            for ext in ['src', 'bkg', 'rsp']:
                src_file = f'{det_dir}/{saved_name}.{ext}'
                dst_file = f'{heapy_tintegrated_path}/tintegrated_{det}.{ext}'
                if os.path.exists(src_file):
                    shutil.copy(src_file, dst_file)
        else:
            print(f"  WARNING: no .src file found for {det} — fit_tintegrated will skip this det")

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