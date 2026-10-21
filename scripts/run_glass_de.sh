#!/bin/bash
set -e
cd "$(dirname "$0")/.."
WORKDIR="$(pwd)"
DATASET="glass"

echo "=========================================================="
echo " Starting DE-SOM Pipeline for: ${DATASET}"
echo "=========================================================="

echo "[Step 1/3] Running 5-fold CV Random Search for DE hyperparameters..."
./ParameterSelection "${DATASET}.dat" "${DATASET}.som.names" 30 "${DATASET}"

echo "[Step 2/3] Running 30 Independent Experiments for DE..."
./Experimental30RunsDE "${DATASET}.dat" "${DATASET}.som.names"

echo "[Step 3/3] Generating Unified Distance Matrix (U-Matrix)..."
python3 -c "
import subprocess, re

with open('results/${DATASET}/random_params_de.txt') as f:
    text = f.read()

np_val = re.search(r'Best NP:\s*(\d+)', text).group(1)
gen_val = re.search(r'Best Gen:\s*(\d+)', text).group(1)
f_val = re.search(r'Best F:\s*([\d\.]+)', text).group(1)
cr_val = re.search(r'Best CR:\s*([\d\.]+)', text).group(1)

subprocess.run(['./generate_de_map', '${DATASET}.dat', '${DATASET}.som.names', '${DATASET}_de.map', '1', np_val, gen_val, cr_val, f_val], check=True)
subprocess.run(['./algorithms/SOM_PAK/umat', '-cin', '${DATASET}_de.map', '-eps', '-border', '-nolabs', '-o', '${DATASET}_de_umat.eps'], check=True)
subprocess.run(['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m', '-r300', '-dEPSCrop', f'-sOutputFile=${DATASET}_de_umat.png', '${DATASET}_de_umat.eps'], check=True)
subprocess.run(['cp', f'${DATASET}_de_umat.png', f'${DATASET}_de_umat.eps', '/home/ubuntu/.gemini/antigravity/brain/3dd05754-d227-45b9-a1eb-9590ef386b14/'], check=True)
print('U-Matrix successfully generated: ${DATASET}_de_umat.png and .eps')
"

echo "=========================================================="
echo " Pipeline Complete for DE-SOM on ${DATASET}!"
echo " Results in: results/${DATASET}/ and workspace root"
echo "=========================================================="
