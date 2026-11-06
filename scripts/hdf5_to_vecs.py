"""Convert ann-benchmarks sift-128-euclidean.hdf5 into TEXMEX-style fvecs/ivecs."""
import os
import sys

import numpy as np


def write_vecs(path, arr, dtype):
    arr = np.ascontiguousarray(arr, dtype=dtype)
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=dtype)
    out[:, 0] = np.array(d, dtype=np.int32).view(dtype)
    out[:, 1:] = arr
    out.tofile(path)


def main(src, dst):
    import h5py  # only needed for conversion

    os.makedirs(dst, exist_ok=True)
    with h5py.File(src, "r") as f:
        write_vecs(os.path.join(dst, "sift_base.fvecs"), f["train"][:], np.float32)
        write_vecs(os.path.join(dst, "sift_query.fvecs"), f["test"][:], np.float32)
        write_vecs(os.path.join(dst, "sift_groundtruth.ivecs"), f["neighbors"][:], np.int32)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
