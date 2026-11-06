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
```

Sanitizer build: `cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DANN_SANITIZE=ON`.

## Layout

```
include/ann/   library headers (dataset.h: fvecs/ivecs reader, 64-byte aligned storage)
tests/         doctest unit tests
tools/         small utilities
scripts/       data download and conversion
third_party/   vendored doctest.h (MIT)
```
