#!/bin/bash
set -e
cd "$(dirname "$0")/.."
WORKDIR="$(pwd)"
DATASET="wine"

echo "=========================================================="
echo " Starting DE-SOM Pipeline for: ${DATASET}"
echo "=========================================================="

echo "[Step 1/3] Running 5-fold CV Random Search for DE hyperparameters..."
./ParameterSelection "${DATASET}.dat" "${DATASET}.som.names" 30 "${DATASET}"

echo "[Step 2/3] Running 30 Independent Experiments for DE..."
./Experimental30RunsDE "${DATASET}.dat" "${DATASET}.som.names"

echo "[Step 3/3] Generating Unified Distance Matrix (U-Matrix)..."
python3 -c "
import subprocess

subprocess.run(['./algorithms/SOM_PAK/umat', '-cin', 'results/${DATASET}/${DATASET}_de.map', '-eps', '-border', '-nolabs', '-o', f'results/${DATASET}/${DATASET}_de_umat.eps'], check=True)
subprocess.run(['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m', '-r300', '-dEPSCrop', f'-sOutputFile=results/${DATASET}/${DATASET}_de_umat.png', f'results/${DATASET}/${DATASET}_de_umat.eps'], check=True)
subprocess.run(['cp', f'results/${DATASET}/${DATASET}_de_umat.png', f'results/${DATASET}/${DATASET}_de_umat.eps', '.'], check=True)
print('U-Matrix successfully generated: results/${DATASET}/${DATASET}_de_umat.png and .eps')
"

echo "=========================================================="
echo " Pipeline Complete for DE-SOM on ${DATASET}."
echo " Results in: results/${DATASET}/ and workspace root"
echo "=========================================================="
