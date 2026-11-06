# ann-engine

HNSW approximate nearest neighbor search from scratch in C++17, with AVX2 distance kernels, benchmarked against hnswlib on SIFT1M.

> Work in progress. Results, plots and the ablation table will be added once benchmarks run.

## Quick start (Linux / WSL2)

```bash
sudo apt install -y build-essential cmake python3-pandas python3-matplotlib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ann_tests                 # unit tests
./scripts/download_sift.sh        # ~500 MB into data/sift/
./build/inspect_data data/sift    # sanity check: shapes and load time
./scripts/run_baselines.sh        # brute force + hnswlib sweep -> results/*.csv, results/recall_qps.png
./scripts/run_hnsw.sh             # our HNSW -> results/hnsw.csv, plot with both
```

Run benchmarks plugged in, with other heavy apps closed. Single-thread runs are pinned with `taskset -c 0`.

Sanitizer build: `cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DANN_SANITIZE=ON`.

## Layout

```
include/ann/   library headers: dataset.h (fvecs/ivecs, 64-byte aligned), distance.h (scalar/AVX2/NEON L2),
               bruteforce.h (exact top-k), hnsw.h (our index), visited.h (epoch visited list),
               metrics.h (recall, percentiles), timer.h
src/           library sources
bench/         benchmark driver (brute, hnswlib, hnsw), CSV output
tests/         doctest unit tests
tools/         small utilities
scripts/       data download and conversion
results/       benchmark CSVs and plots (index caches are git-ignored)
third_party/   vendored doctest.h (MIT) and hnswlib v0.8.0 headers (Apache-2.0), so builds need no network
```
