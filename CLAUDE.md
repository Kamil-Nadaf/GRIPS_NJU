# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Everything runs inside Docker. Build and launch:

```bash
docker build -t grips:latest -f Dockerfile .
docker run -d -p 8888:8888 \
  -v /Users/kamil/Projects/GRIPS_NJU:/workspace \
  -v /Users/kamil/Projects/Data:/workspace/data \
  --name grips grips:latest
```

Jupyter Lab at `http://localhost:8888`. The kernel is registered as "GRIPS (Python 3.12)".

**Important:** The pipeline runs inside Docker, not on macOS directly. `/workspace` is the repo mount; `/workspace/data` is the external data mount. Code edits happen on the Mac side (mounted into `/workspace`), but execution requires the Docker container's Python environment (heapyx, bayspec, pymultinest, MultiNest).

## Architecture

**Pipeline flow** (orchestrated by `GRB_Analysis/GRB_pipeline.ipynb`):

1. `load_grb(grb)` — downloads Fermi GBM TTE + poshist files from FTP, then pushes all path variables (`GRB_Name`, `DATA_BASE`, `sel_dets`, `heapy_tintegrated_path`, `bayspec_tintegrated_path`, etc.) into the caller's global namespace via `inspect.stack()`.
2. `extract_geometry()` — SAA passage, visibility, detector angles, skymap.
3. `extract_tintegrated_spectra()` — per-detector lightcurves (saved as Plotly HTML via heapyx `extract_curve(show=True)`) and count spectra (`.src`/`.bkg`/`.rsp`). Flattens auto-named files from heapyx into fixed names (`tintegrated_{det}.*`).
4. `fit_tintegrated(model_name)` — bayspec Bayesian spectral fitting with MultiNest/pymultinest for nested sampling. Produces posterior plots under `bayspec_tintegrated_path/{model_name}/`.

**Key design pattern:** `load_grb()` uses `inspect.stack()[1].frame.f_globals.update(to_push)` to inject all local variables into the calling notebook's namespace AND into `grb_utils`'s own `sys.modules[__name__].__dict__`. This means all other pipeline functions (`extract_geometry`, `extract_tintegrated_spectra`, `fit_tintegrated`) access variables like `GRB_Name`, `DATA_BASE`, `sel_dets`, `fermi_met`, `gbm_rtv` as module-level globals — they are NOT passed as parameters. This is why the notebook must call `load_grb(grb)` before any other pipeline function.

**Data layout** (under `DATA_BASE = /workspace/data`):

```
{GRB_NAME}/
  data/
    tintegrated/
      heapy/
        tintegrated_{det}.src   # flattened by extract_tintegrated_spectra
        tintegrated_{det}.bkg
        tintegrated_{det}.rsp
        {det}/lc.html           # Plotly lightcurve (parsed by plot_lightcurves.py)
      bayspec/
        {model_name}/           # fit outputs, posterior plots
    tresolved/
      heapy/
      bayspec/
        {GRB_NAME}_bayspec_data.h5
    Sampler_Output/
      {GRB_NAME}_posterior_stat.h5
  geometry/                     # skymap from extract_geometry
gbm_data/                       # raw downloaded TTE .fit.gz files
```

## Key modules

- **`grb_config.py`** — `grbs_df` DataFrame: GRB catalog with `name`, `ra`, `dec`, `utc`, `sel_dets`. Currently active: GRB131011A only.
- **`grb_utils.py`** — All pipeline logic. Depends on `heapyx` (Fermi GBM data retrieval, TTE binning, lightcurve/spectrum extraction) and `bayspec` (Bayesian spectral fitting with MultiNest). The module-level `DATA_BASE = '/workspace/data'` is the root for all GRB data directories.
- **`plot_lightcurves.py`** — Standalone script that parses heapyx-generated Plotly HTML files (`lc.html`) and plots multi-detector lightcurves with matplotlib. Can run standalone (`python plot_lightcurves.py GRB131011A`) or imported in a notebook. Requires Stage 3 (`extract_tintegrated_spectra`) to have completed first.

## After editing .py files

When editing `grb_utils.py` or `plot_lightcurves.py` on the Mac side, changes are visible inside Docker immediately (volume mount). However, the Jupyter kernel caches imported modules — either restart the kernel or:

```python
import importlib
import grb_utils; importlib.reload(grb_utils)
import plot_lightcurves; importlib.reload(plot_lightcurves)
```
