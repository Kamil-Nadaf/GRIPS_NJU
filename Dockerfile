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

# Install system dependencies
RUN for i in 1 2 3 4 5; do \
        apt-get update && \
        apt-get install -y git build-essential cmake gfortran liblapack-dev libblas-dev \
            libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
            libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
            libgbm1 libpango-1.0-0 libcairo2 libasound2 fonts-liberation && \
        break || sleep 10; \
    done && \
    rm -rf /var/lib/apt/lists/*

# Build MultiNest (libmultinest.so) — nested-sampling backend bayspec/pymultinest need.
# Not on PyPI/apt as a compiled lib, must build from source; native ARM64 on M1 automatically.
RUN git clone https://github.com/JohannesBuchner/MultiNest.git /opt/MultiNest && \
    mkdir -p /opt/MultiNest/build && \
    cd /opt/MultiNest/build && \
    cmake .. && \
    make

ENV LD_LIBRARY_PATH="/opt/MultiNest/lib"

# Install heapyx and its dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install heapyx

# Clone and install responsum
RUN git clone https://github.com/jyangch/responsum.git && \
    pip install ./responsum && \
    rm -rf responsum

# Clone and install gbmgeometry
RUN git clone https://github.com/jyangch/gbmgeometry.git && \
    pip install ./gbmgeometry && \
    rm -rf gbmgeometry

# Clone and install gbm_drm_gen
RUN git clone https://github.com/jyangch/gbm_drm_gen.git && \
    pip install ./gbm_drm_gen && \
    rm -rf gbm_drm_gen

# Install bayspec
RUN pip install bayspec

# Python wrapper for MultiNest — bayspec's .multinest() call needs this
RUN pip install pymultinest

# Kaleido (plotly's static-image backend) needs headless Chrome to export
# PNG/PDF from plotly figures — bayspec's Plot.save() calls this internally.
RUN plotly_get_chrome -y

# Install Jupyter and notebooks
RUN pip install jupyter jupyterlab ipython ipykernel

# Install additional scientific packages
RUN pip install arviz pandas scipy matplotlib astropy cartopy bilby

RUN pip install tables ipython_genutils

# Set Jupyter configuration
ENV JUPYTER_ENABLE_LAB=yes

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