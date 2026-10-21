#!/bin/bash
cd "$(dirname "$0")/.."
WORKDIR="$(pwd)"

mkdir -p results/{iris,glass,seeds,haberman,vertebral}

echo "Launching all 10 pipelines (5 datasets x 2 algorithms) in background..."

nohup ./scripts/run_iris_som.sh > results/iris/som.log 2>&1 &
echo "  [1/10] Iris SOM launched (PID: $!). Log: results/iris/som.log"

nohup ./scripts/run_iris_de.sh > results/iris/de.log 2>&1 &
echo "  [2/10] Iris DE launched (PID: $!). Log: results/iris/de.log"

nohup ./scripts/run_glass_som.sh > results/glass/som.log 2>&1 &
echo "  [3/10] Glass SOM launched (PID: $!). Log: results/glass/som.log"

nohup ./scripts/run_glass_de.sh > results/glass/de.log 2>&1 &
echo "  [4/10] Glass DE launched (PID: $!). Log: results/glass/de.log"

nohup ./scripts/run_seeds_som.sh > results/seeds/som.log 2>&1 &
echo "  [5/10] Seeds SOM launched (PID: $!). Log: results/seeds/som.log"

nohup ./scripts/run_seeds_de.sh > results/seeds/de.log 2>&1 &
echo "  [6/10] Seeds DE launched (PID: $!). Log: results/seeds/de.log"

nohup ./scripts/run_haberman_som.sh > results/haberman/som.log 2>&1 &
echo "  [7/10] Haberman SOM launched (PID: $!). Log: results/haberman/som.log"

nohup ./scripts/run_haberman_de.sh > results/haberman/de.log 2>&1 &
echo "  [8/10] Haberman DE launched (PID: $!). Log: results/haberman/de.log"

nohup ./scripts/run_vertebral_som.sh > results/vertebral/som.log 2>&1 &
echo "  [9/10] Vertebral SOM launched (PID: $!). Log: results/vertebral/som.log"

nohup ./scripts/run_vertebral_de.sh > results/vertebral/de.log 2>&1 &
echo "  [10/10] Vertebral DE launched (PID: $!). Log: results/vertebral/de.log"

echo ""
echo "All 10 pipelines running concurrently!"
echo "Inspect any log live via: tail -f results/<dataset>/<som|de>.log"
