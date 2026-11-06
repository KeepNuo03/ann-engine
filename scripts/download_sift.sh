#!/usr/bin/env bash
# Downloads SIFT1M into data/sift/ (sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs).
# Primary source: TEXMEX corpus (FTP). Fallback: ann-benchmarks HDF5 converted to fvecs.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=data
mkdir -p "$DATA"

if [[ -f $DATA/sift/sift_base.fvecs && -f $DATA/sift/sift_query.fvecs && -f $DATA/sift/sift_groundtruth.ivecs ]]; then
  echo "SIFT1M already present in $DATA/sift"
  exit 0
fi

echo "Trying TEXMEX (ftp.irisa.fr)..."
if curl -fL --retry 3 --connect-timeout 20 -o "$DATA/sift.tar.gz" \
     ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz; then
  tar -xzf "$DATA/sift.tar.gz" -C "$DATA"
  rm -f "$DATA/sift.tar.gz"
else
  echo "TEXMEX failed, falling back to ann-benchmarks HDF5 (needs: sudo apt install python3-h5py)"
  curl -fL --retry 3 -o "$DATA/sift-128-euclidean.hdf5" \
    http://ann-benchmarks.com/sift-128-euclidean.hdf5
  python3 scripts/hdf5_to_vecs.py "$DATA/sift-128-euclidean.hdf5" "$DATA/sift"
fi

ls -lh "$DATA/sift"
