#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ann/dataset.h"

namespace ann {

// recall@k of one result list against the first k ground-truth ids.
inline double recall_at_k(const uint32_t* result, size_t n_result, const int32_t* gt, size_t k) {
  size_t hit = 0;
  for (size_t i = 0; i < std::min(n_result, k); ++i)
    for (size_t j = 0; j < k; ++j)
      if (static_cast<int32_t>(result[i]) == gt[j]) {
        ++hit;
        break;
      }
  return static_cast<double>(hit) / static_cast<double>(k);
}

// Mean recall@k over all queries. results is (nq x k) row-major.
inline double mean_recall(const std::vector<uint32_t>& results, const IdMatrix& gt, size_t k) {
  const size_t nq = gt.n();
  double sum = 0;
  for (size_t i = 0; i < nq; ++i) sum += recall_at_k(&results[i * k], k, gt.row(i), k);
  return sum / static_cast<double>(nq);
}

// p-th percentile (0..100) of a copy of the samples.
inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(v.size() - 1) + 0.5);
  std::nth_element(v.begin(), v.begin() + static_cast<long>(idx), v.end());
  return v[idx];
}

}  // namespace ann
