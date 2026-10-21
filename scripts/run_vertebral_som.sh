#!/bin/bash
set -e
cd "$(dirname "$0")/.."
WORKDIR="$(pwd)"
DATASET="vertebral"

echo "=========================================================="
echo " Starting SOM_PAK Pipeline for: ${DATASET}"
echo "=========================================================="

echo "[Step 1/3] Running 5-fold CV Random Search for SOM hyperparameters..."
./ParameterSelectionSOM "${DATASET}.dat" "${DATASET}.som.names"

echo "[Step 2/3] Running 30 Independent Experiments for SOM..."
./Experimental30RunsSOM "${DATASET}.dat" "${DATASET}.som.names"

echo "[Step 3/3] Generating Unified Distance Matrix (U-Matrix)..."
python3 -c "
import subprocess, re

with open('results/${DATASET}/random_params_som.txt') as f:
    text = f.read()

lr = re.search(r'Best Initial LR:\s*([\d\.]+)', text).group(1)
lr_decay = re.search(r'Best LR Decay:\s*([\d\.]+)', text).group(1)
radius = re.search(r'Best Initial Radius:\s*([\d\.]+)', text).group(1)
radius_decay = re.search(r'Best Radius Decay:\s*([\d\.]+)', text).group(1)

raw_data = []
with open('${DATASET}.dat') as f:
    dim = int(f.readline().strip())
    for line in f:
        parts = line.strip().split()
        if parts:
            raw_data.append([float(x) for x in parts[:dim]])

mins = [min(row[j] for row in raw_data) for j in range(dim)]
maxs = [max(row[j] for row in raw_data) for j in range(dim)]
scaled = []
for row in raw_data:
    scaled.append([(row[j] - mins[j]) / (maxs[j] - mins[j] if maxs[j] > mins[j] else 1.0) for j in range(dim)])

with open('${DATASET}_normalized.som', 'w') as f:
    f.write(f'{dim}\n')
    for row in scaled:
        f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

subprocess.run(['./algorithms/SOM_PAK/randinit', '-din', '${DATASET}_normalized.som', '-cout', '${DATASET}_som_init.map', '-topol', 'hexa', '-neigh', 'gaussian', '-xdim', '8', '-ydim', '6', '-rand', '42'], check=True)
subprocess.run(['./algorithms/SOM_PAK/vsom', '-din', '${DATASET}_normalized.som', '-vin', '${DATASET}_normalized.som', '-cin', '${DATASET}_som_init.map', '-cout', '${DATASET}_som.map', '-lrate', lr, '-lratedecay', lr_decay, '-radius', radius, '-radiusdecay', radius_decay, '-rand', '42'], check=True)
subprocess.run(['./algorithms/SOM_PAK/umat', '-cin', '${DATASET}_som.map', '-eps', '-border', '-nolabs', '-o', '${DATASET}_som_umat.eps'], check=True)
subprocess.run(['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m', '-r300', '-dEPSCrop', f'-sOutputFile=${DATASET}_som_umat.png', '${DATASET}_som_umat.eps'], check=True)
subprocess.run(['cp', f'${DATASET}_som_umat.png', f'${DATASET}_som_umat.eps', '/home/ubuntu/.gemini/antigravity/brain/3dd05754-d227-45b9-a1eb-9590ef386b14/'], check=True)
print('U-Matrix successfully generated: ${DATASET}_som_umat.png and .eps')
"

echo "=========================================================="
echo " Pipeline Complete for SOM on ${DATASET}!"
echo " Results in: results/${DATASET}/ and workspace root"
echo "=========================================================="
