#!/bin/bash
# Start Jupyter Lab + one Streamlit UI (time-integrated GRB analysis).
set -uo pipefail

STREAMLIT_PID=""

_cleanup() {
  if [[ -n "${STREAMLIT_PID}" ]] && kill -0 "${STREAMLIT_PID}" 2>/dev/null; then
    kill "${STREAMLIT_PID}" 2>/dev/null || true
  fi
}
trap _cleanup EXIT

_streamlit_loop() {
  while true; do
    pkill -f 'streamlit run ui/app.py' 2>/dev/null || true
    sleep 1
    echo "[entrypoint] starting Streamlit on :8501" >&2
    streamlit run ui/app.py \
      --server.port=8501 \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false &
    STREAMLIT_PID=$!
    wait "${STREAMLIT_PID}"
    code=$?
    echo "[entrypoint] Streamlit exited ($code); restart in 3s" >&2
    sleep 3
  done
}

_streamlit_loop &

exec jupyter lab \
  --ip=0.0.0.0 \
  --allow-root \
  --no-browser \
  --ServerApp.name='GBM' \
  --LabApp.app_version='' \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --notebook-dir=/workspace
