// Exact k-nearest-neighbor search by scanning every base vector.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ann/dataset.h"

namespace ann {

// Top-k for one query, sorted by increasing distance. Returns (distance, id).
std::vector<std::pair<float, uint32_t>> bruteforce_knn(const Dataset& base, const float* q,
                                                       size_t k);

// Top-k ids for every query, written row-major into an (nq x k) matrix.
// Parallel over blocks of queries; within a block, base vectors are scanned in
// chunks so each chunk is reused by all queries of the block while it is in cache.
IdMatrix bruteforce_knn_batch(const Dataset& base, const Dataset& queries, size_t k,
                              int threads);

}  // namespace ann
