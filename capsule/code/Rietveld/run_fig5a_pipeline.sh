#!/usr/bin/env bash
set -euo pipefail

PT_PATH=$1
PRMFILE=$2
OUT_DIR=$3
MULTI_FLAG="${4:-}"

mkdir -p "${OUT_DIR}"

echo "===== STEP 1: prepare cif (PXRDGen env) ====="

python fig5a_prepare_plot.py prepare \
  --pt_path "${PT_PATH}" \
  --out_dir "${OUT_DIR}" \
  ${MULTI_FLAG}

echo "===== STEP 2: GSAS refinement (GSAS Python) ====="

/workspace/g2full/bin/python fig5a_rietveld_gsas.py \
  --manifest "${OUT_DIR}/fig5a_manifest.csv" \
  --prmfile "${PRMFILE}" \
  --out_dir "${OUT_DIR}"

echo "===== STEP 3: plot (PXRDGen env) ====="

python fig5a_prepare_plot.py plot \
  --results "${OUT_DIR}/fig5a_results.csv" \
  --out_dir "${OUT_DIR}"

echo "Pipeline finished."