#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_conda_env.sh [env_name]

ENV_NAME="${1:-pxrdgen}"
PYTHON_VERSION="3.9"

# =========================
# GSAS-II configuration
# =========================

GSAS2_INSTALL_DIR="${GSAS2_INSTALL_DIR:-/workspace/g2full}"
GSAS2_URL="${GSAS2_URL:-}"
SKIP_GSAS2="${SKIP_GSAS2:-0}"
SKIP_GSAS2_BINARY_BUILD="${SKIP_GSAS2_BINARY_BUILD:-0}"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  if [[ -x "/workspace/miniforge/bin/conda" ]]; then
    export PATH="/workspace/miniforge/bin:${PATH}"
  elif [[ -x "${HOME}/miniforge/bin/conda" ]]; then
    export PATH="${HOME}/miniforge/bin:${PATH}"
  elif [[ -x "/opt/conda/bin/conda" ]]; then
    export PATH="/opt/conda/bin:${PATH}"
  else
    echo "Error: conda not found in PATH."
    exit 1
  fi
}

find_conda

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Error: this setup script is currently tested only on Linux."
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Error: this setup script expects x86_64 binaries."
  exit 1
fi

echo "[1/5] Creating conda environment: ${ENV_NAME} (python=${PYTHON_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment ${ENV_NAME} already exists; reusing it."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "[2/5] Upgrading pip tooling"
conda run -n "${ENV_NAME}" python -m pip install -U pip wheel "setuptools<81"

echo "[3/5] Installing Python dependencies"
conda run -n "${ENV_NAME}" python -m pip install -U "numpy<2"

echo "Installing PyTorch CUDA 11.8 wheels"
conda run -n "${ENV_NAME}" python -m pip install -U \
  torch==2.1.0+cu118 \
  torchaudio==2.1.0+cu118 \
  torchvision==0.16.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

echo "Installing scipy for PyTorch Geometric extension dependencies"
conda run -n "${ENV_NAME}" python -m pip install -U scipy

echo "Installing PyTorch Geometric extension wheels"
conda run -n "${ENV_NAME}" python -m pip install -U \
  torch-cluster==1.6.3 \
  torch-scatter==2.1.2 \
  torch-sparse==0.6.18 \
  torch-spline-conv==1.2.2 \
  --no-index \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

echo "Installing remaining Python dependencies"
conda run -n "${ENV_NAME}" python -m pip install -U \
  hydra-core==1.3.2 \
  torch_geometric==2.4.0 \
  lightning==2.1.4 \
  torchmetrics==1.3.0.post0 \
  matminer==0.9.0 \
  pymatgen==2023.8.10 \
  SMACT==2.5.5 \
  fastdtw==0.3.4 \
  tqdm==4.66.1 \
  p_tqdm==1.4.0 \
  chemparse==0.3.1 \
  pandas \
  matplotlib \
  wandb

echo "Installing build tools for GSAS-II Python ${PYTHON_VERSION} extensions"
conda install -y -n "${ENV_NAME}" -c conda-forge meson ninja cython pkg-config compilers

echo "[4/5] Installing GSAS-II to ${GSAS2_INSTALL_DIR}"
if [[ "${SKIP_GSAS2}" == "1" ]]; then
  echo "Skipping GSAS-II install"
  exit 0
fi

TMP_G2_SCRIPT="/tmp/gsas2_installer.sh"

try_download() {
  local url="$1"

  if [[ -z "${url}" ]]; then
    return 1
  fi

  echo "Trying GSAS-II URL: ${url}"

  if ! curl -fL "${url}" -o "${TMP_G2_SCRIPT}"; then
    return 1
  fi

  if [[ ! -s "${TMP_G2_SCRIPT}" ]] || ! head -n 1 "${TMP_G2_SCRIPT}" | grep -q '^#!'; then
    return 1
  fi

  chmod +x "${TMP_G2_SCRIPT}"
  return 0
}

DOWNLOADED=0

if [[ -n "${GSAS2_URL}" ]]; then
  if try_download "${GSAS2_URL}"; then
    DOWNLOADED=1
  fi
else
  CANDIDATE_URLS=(
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/gsas2full-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/latest/download/gsas2full-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/latest/download/gsas2main-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/latest/download/gsas2main-rhel-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/gsas2main-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/gsas2main-rhel-Latest-Linux-x86_64.sh"
  )

  for url in "${CANDIDATE_URLS[@]}"; do
    if try_download "${url}"; then
      GSAS2_URL="${url}"
      DOWNLOADED=1
      break
    fi
  done
fi

if [[ "${DOWNLOADED}" != "1" ]]; then
  echo "Failed to download GSAS-II installer"
  exit 1
fi

echo "Using installer: ${GSAS2_URL}"
bash "${TMP_G2_SCRIPT}" -b -p "${GSAS2_INSTALL_DIR}"

echo "[5/5] Building GSAS-II binary extensions for ${ENV_NAME}"
if [[ "${SKIP_GSAS2_BINARY_BUILD}" == "1" ]]; then
  echo "Skipping GSAS-II binary extension build"
else
  GSAS2_SRC="${GSAS2_INSTALL_DIR}/GSAS-II"
  if [[ ! -f "${GSAS2_SRC}/meson.build" ]]; then
    echo "Error: GSAS-II source tree not found at ${GSAS2_SRC}"
    exit 1
  fi

  GSAS2_SRC="${GSAS2_SRC}" python3 - <<'PATCH_GSAS_MESON'
from pathlib import Path
import os
root = Path(os.environ['GSAS2_SRC'])
for rel in ['sources/meson.build', 'sources/k_vec_cython/meson.build']:
    path = root / rel
    text = path.read_text()
    path.write_text(text.replace('numpy_dep', 'np_dep'))
PATCH_GSAS_MESON

  BUILD_DIR="${GSAS2_SRC}/build_py39"
  conda run -n "${ENV_NAME}" bash -lc "cd '${GSAS2_SRC}' && rm -rf '${BUILD_DIR}' && meson setup '${BUILD_DIR}' --prefix='${GSAS2_INSTALL_DIR}/gsasii-py39-build' && meson compile -C '${BUILD_DIR}'"

  PY_SUFFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(f"p{sys.version_info[0]}.{sys.version_info[1]}")')"
  NP_SUFFIX="$(conda run -n "${ENV_NAME}" python -c 'import numpy as np; v=np.__version__.split("."); print(f"n{v[0]}.{v[1]}")')"
  BIN_DIR="${GSAS2_SRC}/GSASII-bin/linux_64_${PY_SUFFIX}_${NP_SUFFIX}"
  mkdir -p "${BIN_DIR}"
  cp "${BUILD_DIR}"/sources/*.so "${BUILD_DIR}"/sources/LATTIC "${BUILD_DIR}"/sources/convcell "${BUILD_DIR}"/sources/GSASIIversion.txt "${BIN_DIR}/"
  cp "${BUILD_DIR}"/sources/k_vec_cython/*.so "${BIN_DIR}/"
  echo "Installed GSAS-II Python extensions to ${BIN_DIR}"
fi

cat <<EOF

Setup complete.

Activate env:
  conda activate ${ENV_NAME}

Recommended GSAS-II paths for this repository:
  export PYTHONPATH=${GSAS2_INSTALL_DIR}/GSAS-II/backcompat:${GSAS2_INSTALL_DIR}/GSAS-II/GSASII:${GSAS2_INSTALL_DIR}/GSAS-II:\$PYTHONPATH

GSAS-II path:
  ${GSAS2_INSTALL_DIR}
EOF
