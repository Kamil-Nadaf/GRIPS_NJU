# Fermi GBM GRB Analysis

Time-integrated Fermi GBM spectral fitting with **heapy** (data + lightcurves + count spectra) and **bayspec** (Bayesian nested sampling). Optional **3ML** cross-check and GCN comparison.

Everything runs in Docker. Do not use the host macOS Python for fits.

## Quick start

### 1. Clone

```bash
git clone git@github.com:Kamil-Nadaf/fermi-gbm-analysis.git
cd fermi-gbm-analysis
```

### 2. Set host paths

Bind mounts need **absolute** host paths. If you move or rename the repo later, recreate the container so mounts stay valid.

```bash
# This repo (code) → /workspace
export REPO_ROOT="$(pwd)"

# Your GBM data tree (TTE downloads + fit products) → /workspace/data
export DATA_HOST=/absolute/path/to/Data
```

### 3. Build image and run container

**Option A — Docker Compose (recommended)**

```bash
docker compose up --build -d
```

**Option B — `docker build` / `docker run` (same image/container names)**

```bash
docker build -t gbm:latest -f Dockerfile .

docker stop gbm 2>/dev/null || true
docker rm gbm 2>/dev/null || true

docker run -d \
  -p 8888:8888 -p 8501:8501 \
  -v "${REPO_ROOT}:/workspace" \
  -v "${DATA_HOST}:/workspace/data" \
  -e DATA_BASE=/workspace/data \
  -e MPLBACKEND=Agg \
  -e OMP_NUM_THREADS=1 \
  -e PYTHONPATH=/workspace \
  --name gbm gbm:latest
```

| Service | URL |
|---------|-----|
| Jupyter Lab | http://localhost:8888 — kernel **GBM (Python 3.12)** |
| Streamlit UI | http://localhost:8501 — time-integrated only |

Restart / rebuild:

```bash
docker compose down
docker compose up --build -d
# or re-run the Option B stop/rm/build/run block
```

### 4. Smoke-test the pipeline

```bash
# Unit tests
docker exec gbm python -m unittest discover -s /workspace/tests -v

# Short tint fit (nlive=200); needs TTE data under $DATA_HOST
docker exec gbm python -m cli.run_grb GRB150514A --nlive 200 --workers 1
```

Then open the Streamlit UI or `GRB_pipeline.ipynb` in Jupyter.

## What the UI does

1. Pick a catalog GRB (default prefers bursts with GCN refs)
2. Set tint window `t1`–`t2`, model (`cpl` default), NaI-only or +BGO
3. **Run time integrated** → geometry → heapy spectra → bayspec fit → HDF5 params
4. Optionally also run **3ML** on the same tint window
5. Inspect lightcurve, fit plots, parameters, and **GCN · bayspec · 3ML** table

Time-resolved fitting is **not** in the UI (CLI / notebook only, opt-in).

## CLI (inside the container)

```bash
# Tint pipeline (geometry + spectra + bayspec + HDF5)
docker exec gbm python -m cli.run_grb GRB150514A --nlive 1000 --workers 1

# Force overwrite fingerprinted products
docker exec gbm python -m cli.run_grb GRB150514A --force --workers 1

# Optional 3ML tint
docker exec gbm python -m cli.run_3ml GRB150514A --nlive 1000 --force

# Catalog / models
docker exec gbm python -m cli.run_grb --list
docker exec gbm python -m cli.run_grb --list-models
```

`--nlive 200` for smoke tests; `1000` for production. Default spectral fits are **NaI only**; add `--include-bgo` when needed. Prefer `--workers 1` for heapy extraction.

## Notebook

Open `GRB_pipeline.ipynb` with the **GBM (Python 3.12)** kernel.

Tint-first: load → geometry → heapy tint → bayspec → 3ML → GCN comparison. After editing `.py` files, restart the kernel or `importlib.reload`.

## Library API

```python
from pipeline.runner import GRBPipelineRunner
from pipeline.util import list_grb_results, view_hdf5

runner = GRBPipelineRunner(
    n_workers=1, model_name='cpl', nlive=1000, include_bgo=False)
ctx = runner.run('GRB150514A')  # geometry, spectra_tint, fit_tint, params
list_grb_results(ctx.name)
view_hdf5(ctx.paths.bayspec_data)
```

Override data root with `DATA_BASE` (env) or `GRBContext.from_name(name, data_base=...)`.

## Catalog & GCN cross-checks

Pulse catalog (detectors + slice boundaries) comes from Yan et al. 2024 Table C1
([ApJ 962:85](https://doi.org/10.3847/1538-4357/ad14fb); arXiv: [2308.00772](https://arxiv.org/abs/2308.00772)).
Tint windows `t1`/`t2` for a few bursts follow published GCN intervals (may differ from Table C1 slices, which stay for later time-resolved work).

The UI **GCN compare** table shows these **literature** values next to your fit — they are not results from this repo. Always check the circulars:

| GRB | Literature source | Tint (s) | α | Ep (keV) |
|-----|-------------------|----------|---|----------|
| GRB140606B | [GCN 16363](https://gcn.nasa.gov/circulars/16363) CPL | −3.0 … 12.3 | −1.22 ± 0.04 | 473 ± 83 |
| GRB150514A | [GCN 17819](https://gcn.nasa.gov/circulars/17819) Band\* | 0 … 11.3 | −1.34 ± 0.07 | 73 ± 6 |
| GRB190829A | [GCN 25575](https://gcn.nasa.gov/circulars/25575) CPL | 0 … 4.0 | −1.41 ± 0.08 | 130 ± 20 |

\*GBM published Band in GCN 17819; this pipeline fits CPL by default (soft Ep + steep β → roughly comparable). Optional Konus-Wind check: [GCN 17823](https://gcn.nasa.gov/circulars/17823) CPL.

Do **not** use GRB 221009A (BOAT) for GCN CPL validation — the main episode saturates GBM.

Geometry still uses catalog detectors (may include BGO). Spectral fits default to NaI.

## Data layout

Under `DATA_BASE` (Docker mount, not in this repo):

```
{GRB}/
  geometry/
  data/
    tintegrated/
      heapy/          # LC + .src/.bkg/.rsp
      bayspec/{model}/
      3ML/{model}/
    tresolved/
      bayspec/{GRB}_bayspec_data.h5   # tint_* keys
      3ML/{GRB}_3ML_data.h5
```

Fingerprinted outputs are skipped unless `--force` / `force=True`. Changing `utc`, `t1`, or `t2` changes the fingerprint.

## Tests

```bash
docker exec gbm python -m unittest discover -s /workspace/tests -v
```

## Layout

```
.
  Dockerfile
  docker-compose.yml
  entrypoint.sh          # Jupyter + Streamlit
  README.md
  pipeline/              # library (context, runner, heapy, bayspec, 3ML)
  plotting/
  ui/                    # Streamlit (tint only)
  cli/                   # run_grb, run_3ml
  tests/
  grb_config.py          # catalog + GCN_REFS
  grb_utils.py           # notebook shim
  plot_lightcurves.py
  GRB_pipeline.ipynb
```

## Environment variables

| Variable | Meaning |
|----------|---------|
| `DATA_BASE` | In-container data root (default `/workspace/data`) |
| `DATA_HOST` | Host path mounted at `/workspace/data` |
| `REPO_ROOT` | Host repo path mounted at `/workspace` (default: `.`) |
| `GRB_ACTIVE` | Default active GRB(s) for CLI when no name given |
| `GRB_N_WORKERS` | Default CPU workers |

## Production notes

- Run fits in the `gbm` container only (MultiNest / heapy / 3ML).
- Default: one model per run (`cpl`), NaI-only, tint stages only.
- Prefer `workers=1` for heapy extraction.
- Set `OMP_NUM_THREADS=1` (compose already does) for nested sampling.
- After code edits on the host, they appear via the volume mount; restart Streamlit/Jupyter if imports are cached.
- Switch UI Light/Dark via **☰ → Settings → Theme**.
