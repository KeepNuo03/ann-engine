#!/usr/bin/env bash
# CPU profile of single-thread HNSW queries with perf (Linux / WSL2).
#   sudo apt install -y linux-tools-common linux-tools-generic
# On WSL2 the kernel-specific perf package does not exist; the generic one
# works for sampling (perf record). Hardware cache counters are often not
# exposed under WSL2; use Intel VTune on Windows for those.
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/sift
PERF=$(command -v perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
[[ -x "$PERF" ]] || { echo "perf not found: sudo apt install -y linux-tools-common linux-tools-generic"; exit 1; }
[[ -f results/hnsw_M16_efc200.bin ]] || { echo "run scripts/run_hnsw.sh first"; exit 1; }
cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -g" >/dev/null
cmake --build build-prof -j >/dev/null
COMMON="--base $D/sift_base.fvecs --query $D/sift_query.fvecs --gt $D/sift_groundtruth.ivecs --k 10 --algo hnsw --ef 80 --threads 1 --repeats 3 --index-cache results/hnsw_M16_efc200.bin"
for v in v1 v3; do
  echo "== perf stat, variant $v"
  taskset -c 0 "$PERF" stat -e task-clock,cycles,instructions,cache-references,cache-misses \
    ./build-prof/bench $COMMON --variant "$v" 2>&1 | tee "results/perf_stat_$v.txt" | grep -vE "^loaded|^hnsw:"
  echo "== perf record, variant $v"
  taskset -c 0 "$PERF" record -g -o "results/perf_$v.data" ./build-prof/bench $COMMON --variant "$v" >/dev/null 2>&1
  "$PERF" report -i "results/perf_$v.data" --stdio --no-children --percent-limit 2 2>/dev/null \
    | grep -E "^ +[0-9]" | head -15 | tee "results/perf_top_$v.txt"
done
echo "Note: the profile includes loading the dataset and index; the query loop is what matters."
