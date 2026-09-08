# Fermi GBM analysis env + Streamlit UI (single image)
#
# Build from this directory:
#   docker compose up --build -d
#
# Or:
#   docker build -t gbm:latest .
#   docker run -d -p 8888:8888 -p 8501:8501 \
#     -v "$(cd .. && pwd)":/workspace \
#     -v "${DATA_HOST:-$HOME/Data}":/workspace/data \
#     -e DATA_BASE=/workspace/data \
#     --name gbm gbm:latest
#
# Jupyter: http://localhost:8888
# UI:      http://localhost:8501

FROM python:3.12-slim

WORKDIR /workspace

# System dependencies (build tools + MultiNest + plotly chrome + 3ML)
RUN apt-get update && \
    apt-get install -y \
        git build-essential swig gfortran cmake \
        libopenblas-dev liblapack-dev \
        libgomp1 libcfitsio-dev \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
        libgbm1 libpango-1.0-0 libcairo2 libasound2 fonts-liberation && \
    rm -rf /var/lib/apt/lists/*

# heapyx stack
RUN pip install --upgrade pip setuptools wheel && \
    pip install heapyx

RUN git clone https://github.com/jyangch/responsum.git && \
    pip install ./responsum && \
    rm -rf responsum

RUN git clone https://github.com/jyangch/gbmgeometry.git && \
    pip install ./gbmgeometry && \
    rm -rf gbmgeometry

RUN git clone https://github.com/jyangch/gbm_drm_gen.git && \
    pip install ./gbm_drm_gen && \
    rm -rf gbm_drm_gen

RUN pip install bayspec

# Jupyter + Streamlit UI
RUN pip install jupyter jupyterlab ipython ipykernel streamlit

# Scientific extras
RUN pip install arviz pandas scipy matplotlib tables astropy cartopy bilby \
        ipython_genutils pytest dynesty corner

# MultiNest + PyMultiNest (bayspec nested sampling)
RUN git clone https://github.com/JohannesBuchner/MultiNest.git && \
    cd MultiNest && \
    mkdir -p build && cd build && \
    cmake .. && \
    make && \
    cp ../lib/libmultinest.so /usr/local/lib/ && \
    echo '/usr/local/lib' > /etc/ld.so.conf.d/multinest.conf && \
    ldconfig && \
    cd /workspace && rm -rf /workspace/MultiNest

RUN pip install pymultinest

# Kaleido / plotly static export (bayspec Plot.save)
RUN plotly_get_chrome -y

# Optional alternate spectral backend (3ML)
# https://threeml.readthedocs.io/en/stable/notebooks/grb080916C.html
RUN pip install "astromodels>=2.4" "threeml>=2.4"

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV DATA_BASE=/workspace/data
ENV MPLBACKEND=Agg
ENV JUPYTER_ENABLE_LAB=yes
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

RUN python -m ipykernel install --name gbm --display-name "GBM (Python 3.12)"

COPY entrypoint.sh /usr/local/bin/gbm-entrypoint.sh
RUN chmod +x /usr/local/bin/gbm-entrypoint.sh

EXPOSE 8888
EXPOSE 8501

CMD ["/usr/local/bin/gbm-entrypoint.sh"]
