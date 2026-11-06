#!/usr/bin/env bash
# Step 3: our HNSW on SIFT1M (M=16, efc=200), same protocol as run_baselines.sh,
# then one plot with both implementations.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/sift
mkdir -p results
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j >/dev/null
COMMON="--base $D/sift_base.fvecs --query $D/sift_query.fvecs --gt $D/sift_groundtruth.ivecs --k 10"
NPROC=$(nproc)

echo "== hnsw (ours) M=16 efc=200, all $NPROC threads (builds the index unpinned and caches it)"
./build/bench $COMMON --algo hnsw --M 16 --efc 200 \
  --ef 10,20,40,80,120,200,400,800 --threads "$NPROC" --build-threads "$NPROC" \
  --index-cache results/hnsw_M16_efc200.bin --out results/hnsw.csv

echo "== hnsw (ours) M=16 efc=200, single thread (pinned to CPU 0, loads the cached index)"
taskset -c 0 ./build/bench $COMMON --algo hnsw --M 16 --efc 200 \
  --ef 10,20,40,80,120,200,400,800 --threads 1 \
  --index-cache results/hnsw_M16_efc200.bin --out results/hnsw.csv

python3 scripts/plot.py results/hnswlib.csv results/hnsw.csv --out results/recall_qps.png
python3 scripts/plot.py results/hnswlib.csv results/hnsw.csv --threads "$NPROC" --out results/recall_qps_mt.png
