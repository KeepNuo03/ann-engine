#!/usr/bin/env bash
# Step 2 baselines on SIFT1M: exact brute force and hnswlib (M=16, efc=200).
# Single-thread runs are pinned to CPU 0 (a performance core on hybrid Intel CPUs).
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/sift
mkdir -p results
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j >/dev/null
COMMON="--base $D/sift_base.fvecs --query $D/sift_query.fvecs --gt $D/sift_groundtruth.ivecs --k 10"
NPROC=$(nproc)

echo "== brute force, all $NPROC threads (exact reference, should be recall 1.0)"
./build/bench $COMMON --algo brute --threads "$NPROC" --out results/brute.csv

echo "== hnswlib M=16 efc=200, single thread (pinned to CPU 0)"
taskset -c 0 ./build/bench $COMMON --algo hnswlib --M 16 --efc 200 \
  --ef 10,20,40,80,120,200,400,800 --threads 1 --build-threads "$NPROC" \
  --index-cache results/hnswlib_M16_efc200.bin --out results/hnswlib.csv

echo "== hnswlib M=16 efc=200, all $NPROC threads"
./build/bench $COMMON --algo hnswlib --M 16 --efc 200 \
  --ef 10,20,40,80,120,200,400,800 --threads "$NPROC" \
  --index-cache results/hnswlib_M16_efc200.bin --out results/hnswlib.csv

python3 scripts/plot.py results/hnswlib.csv --out results/recall_qps.png
