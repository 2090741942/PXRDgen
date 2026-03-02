#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_conda_env.sh [env_name]
# Example:
#   bash setup_conda_env.sh capsule-py39-cu118

ENV_NAME="${1:-capsule-py39-cu118}"
PYTHON_VERSION="3.9"
GSAS2_INSTALL_DIR="${GSAS2_INSTALL_DIR:-$HOME/g2full}"
GSAS2_URL="${GSAS2_URL:-}"
SKIP_GSAS2="${SKIP_GSAS2:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Please install Miniconda/Anaconda first."
  exit 1
fi

echo "[1/4] Creating conda environment: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

echo "[2/4] Upgrading pip tooling"
conda run -n "${ENV_NAME}" python -m pip install -U pip wheel "setuptools<81"

echo "[3/4] Installing Python dependencies"
# Torch 2.1.0/pyg wheels in this stack are built against NumPy 1.x ABI.
conda run -n "${ENV_NAME}" python -m pip install -U "numpy<2"

conda run -n "${ENV_NAME}" python -m pip install -U \
  hydra-core==1.3.2 \
  torch==2.1.0+cu118 \
  torchaudio==2.1.0+cu118 \
  torchvision==0.16.0+cu118 \
  torch-cluster==1.6.3 \
  torch-scatter==2.1.2 \
  torch-sparse==0.6.18 \
  torch-spline-conv==1.2.2 \
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
  -f https://download.pytorch.org/whl/torch_stable.html \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

echo "[4/4] Installing GSAS-II to ${GSAS2_INSTALL_DIR}"
if [[ "${SKIP_GSAS2}" == "1" ]]; then
  echo "Skipping GSAS-II install because SKIP_GSAS2=1"
  echo
  echo "Done."
  echo "Activate env: conda activate ${ENV_NAME}"
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
  # Avoid executing HTML/404 text.
  if [[ ! -s "${TMP_G2_SCRIPT}" ]] || ! head -n 1 "${TMP_G2_SCRIPT}" | grep -q '^#!'; then
    return 1
  fi
  return 0
}

DOWNLOADED=0
if [[ -n "${GSAS2_URL}" ]]; then
  if try_download "${GSAS2_URL}"; then
    DOWNLOADED=1
  fi
else
  CANDIDATE_URLS=(
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/latest/download/gsas2main-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/latest/download/gsas2main-rhel-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/gsas2main-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/gsas2main-rhel-Latest-Linux-x86_64.sh"
    "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/gsas2full-Latest-Linux-x86_64.sh"
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
  echo "Error: failed to download a valid GSAS-II installer."
  echo "Please set GSAS2_URL explicitly from the release asset page, then rerun:"
  echo "  GSAS2_URL='https://.../gsas2main-<version>-<platform>.sh' bash $0 ${ENV_NAME}"
  exit 1
fi

echo "Using GSAS-II installer: ${GSAS2_URL}"

bash "${TMP_G2_SCRIPT}" -b -p "${GSAS2_INSTALL_DIR}"

echo
echo "Done."
echo "Activate env: conda activate ${ENV_NAME}"
echo "GSAS-II path: ${GSAS2_INSTALL_DIR}"
