#!/usr/bin/env bash
# Step 4 ablation: the same HNSW graph queried with v0..v3, plus hnswlib, all
# single thread pinned to CPU 0. Needs the cached indexes from run_hnsw.sh and
# run_baselines.sh (they are rebuilt with all cores if missing).
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/sift
mkdir -p results
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j >/dev/null
COMMON="--base $D/sift_base.fvecs --query $D/sift_query.fvecs --gt $D/sift_groundtruth.ivecs --k 10"
NPROC=$(nproc)
EF=${EF:-40,80,120,200}

# Make sure both graphs exist before pinning (an index build under taskset would use one core).
[[ -f results/hnsw_M16_efc200.bin ]] || ./build/bench $COMMON --algo hnsw --M 16 --efc 200 \
  --ef 80 --threads "$NPROC" --build-threads "$NPROC" --index-cache results/hnsw_M16_efc200.bin
[[ -f results/hnswlib_M16_efc200.bin ]] || ./build/bench $COMMON --algo hnswlib --M 16 --efc 200 \
  --ef 80 --threads "$NPROC" --build-threads "$NPROC" --index-cache results/hnswlib_M16_efc200.bin

rm -f results/ablation.csv
for v in v0 v1 v2 v3; do
  echo "== hnsw $v"
  taskset -c 0 ./build/bench $COMMON --algo hnsw --variant "$v" --M 16 --efc 200 --ef "$EF" \
    --threads 1 --repeats 5 --index-cache results/hnsw_M16_efc200.bin --out results/ablation.csv
done
echo "== hnswlib"
taskset -c 0 ./build/bench $COMMON --algo hnswlib --M 16 --efc 200 --ef "$EF" \
  --threads 1 --repeats 5 --index-cache results/hnswlib_M16_efc200.bin --out results/ablation.csv

python3 scripts/summarize.py
