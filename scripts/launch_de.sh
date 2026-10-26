#!/bin/bash
cd "$(dirname "$0")/.."
WORKDIR="$(pwd)"

mkdir -p results/{iris,glass,seeds,haberman,vertebral}

echo "Launching 5 DE pipelines concurrently in background..."

nohup ./scripts/run_iris_de.sh > results/iris/de.log 2>&1 &
echo "  [1/5] Iris DE launched (PID: $!). Log: results/iris/de.log"

nohup ./scripts/run_glass_de.sh > results/glass/de.log 2>&1 &
echo "  [2/5] Glass DE launched (PID: $!). Log: results/glass/de.log"

nohup ./scripts/run_seeds_de.sh > results/seeds/de.log 2>&1 &
echo "  [3/5] Seeds DE launched (PID: $!). Log: results/seeds/de.log"

nohup ./scripts/run_haberman_de.sh > results/haberman/de.log 2>&1 &
echo "  [4/5] Haberman DE launched (PID: $!). Log: results/haberman/de.log"

nohup ./scripts/run_vertebral_de.sh > results/vertebral/de.log 2>&1 &
echo "  [5/5] Vertebral DE launched (PID: $!). Log: results/vertebral/de.log"

echo ""
echo "All 5 DE pipelines running concurrently!"
echo "Inspect any log live via: tail -f results/<dataset>/de.log"
echo "Or tail all simultaneously via: tail -f results/*/de.log"
