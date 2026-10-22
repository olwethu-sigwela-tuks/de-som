#!/bin/bash
set -e
cd "$(dirname "$0")/.."
WORKDIR="$(pwd)"
DATASET="glass"

echo "=========================================================="
echo " Starting SOM_PAK Pipeline for: ${DATASET}"
echo "=========================================================="

echo "[Step 1/3] Running 5-fold CV Random Search for SOM hyperparameters..."
./ParameterSelectionSOM "${DATASET}.dat" "${DATASET}.som.names"

echo "[Step 2/3] Running 30 Independent Experiments for SOM..."
./Experimental30RunsSOM "${DATASET}.dat" "${DATASET}.som.names"

echo "[Step 3/3] Generating Unified Distance Matrix (U-Matrix)..."
python3 -c "
import subprocess

subprocess.run(['./algorithms/SOM_PAK/umat', '-cin', 'results/${DATASET}/${DATASET}_som.map', '-eps', '-border', '-nolabs', '-o', f'results/${DATASET}/${DATASET}_som_umat.eps'], check=True)
subprocess.run(['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m', '-r300', '-dEPSCrop', f'-sOutputFile=results/${DATASET}/${DATASET}_som_umat.png', f'results/${DATASET}/${DATASET}_som_umat.eps'], check=True)
subprocess.run(['cp', f'results/${DATASET}/${DATASET}_som_umat.png', f'results/${DATASET}/${DATASET}_som_umat.eps', '.'], check=True)
print('U-Matrix successfully generated: results/${DATASET}/${DATASET}_som_umat.png and .eps')
"

echo "=========================================================="
echo " Pipeline Complete for SOM on ${DATASET}!"
echo " Results in: results/${DATASET}/ and workspace root"
echo "=========================================================="
