# GRIPS — GRB FRED Classifier Environment
# MacBook Pro M1 (automatic ARM64 native)
#
# Build:
#   docker build -t grips:latest -f Dockerfiles/Dockerfile .
#
# Run:
#   docker run -d -p 8888:8888 \
#     -v /Users/kamil/Projects/GRIPS_NJU:/workspace \
#     -v /Users/kamil/Projects/Data:/workspace/data \
#     --name grips grips:latest
#
# Open in browser: http://localhost:8888

FROM python:3.12-slim

WORKDIR /workspace

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Python packages for FRED classifier pipeline
RUN pip install --upgrade pip && \
    pip install \
        numpy \
        scipy \
        matplotlib \
        astropy \
        pandas \
        scikit-learn \
        joblib \
        requests \
        tqdm \
        ipywidgets \
        jupyter \
        jupyterlab \
        ipykernel \
        astro-gdt-fermi

# Initialize GDT data directories (required after install)
RUN gdt-data init

# Official Fermi GBM Data Tools — proper catalog queries + FTP downloads
# (replaces hand-rolled requests calls, handles HEASARC quirks/retries)
RUN pip install astro-gdt astro-gdt-fermi && \
    gdt-data init

# Register kernel as "GRIPS"
RUN python -m ipykernel install --name grips --display-name "GRIPS (Python 3.12)"

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--allow-root", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/workspace"]
