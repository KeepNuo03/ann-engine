#include "ann/bruteforce.h"

#include <algorithm>
#include <queue>

#include "ann/distance.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ann {

namespace {

using Pair = std::pair<float, uint32_t>;  // max-heap on distance by default ordering

// Keeps the k smallest (distance, id) pairs seen so far.
class TopK {
 public:
  explicit TopK(size_t k) : k_(k) { heap_.reserve(k + 1); }
  void push(float dist, uint32_t id) {
    if (heap_.size() < k_) {
      heap_.emplace_back(dist, id);
      std::push_heap(heap_.begin(), heap_.end());
    } else if (dist < heap_.front().first) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = {dist, id};
      std::push_heap(heap_.begin(), heap_.end());
    }
  }
  std::vector<Pair> sorted() const {
    std::vector<Pair> out = heap_;
    std::sort(out.begin(), out.end());
    return out;
  }

 private:
  size_t k_;
  std::vector<Pair> heap_;
};

}  // namespace

std::vector<std::pair<float, uint32_t>> bruteforce_knn(const Dataset& base, const float* q,
                                                       size_t k) {
  TopK top(k);
  const size_t d = base.d();
  for (size_t i = 0; i < base.n(); ++i) top.push(l2sq(q, base.row(i), d), static_cast<uint32_t>(i));
  return top.sorted();
}

IdMatrix bruteforce_knn_batch(const Dataset& base, const Dataset& queries, size_t k,
                              int threads) {
  constexpr size_t kQueryBlock = 16;
  constexpr size_t kBaseChunk = 1024;  // 1024 * 128 * 4 B = 512 KB
  const size_t nq = queries.n(), nb = base.n(), d = base.d();
  const size_t nblocks = (nq + kQueryBlock - 1) / kQueryBlock;
  IdMatrix out(nq, k);
  (void)threads;

#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (long blk = 0; blk < static_cast<long>(nblocks); ++blk) {
    const size_t q0 = static_cast<size_t>(blk) * kQueryBlock;
    const size_t q1 = std::min(nq, q0 + kQueryBlock);
    std::vector<TopK> tops(q1 - q0, TopK(k));
    for (size_t b0 = 0; b0 < nb; b0 += kBaseChunk) {
      const size_t b1 = std::min(nb, b0 + kBaseChunk);
      for (size_t qi = q0; qi < q1; ++qi) {
        const float* q = queries.row(qi);
        TopK& top = tops[qi - q0];
        for (size_t i = b0; i < b1; ++i) top.push(l2sq(q, base.row(i), d), static_cast<uint32_t>(i));
      }
    }
    for (size_t qi = q0; qi < q1; ++qi) {
      auto res = tops[qi - q0].sorted();
      for (size_t j = 0; j < k; ++j)
        out.row(qi)[j] = j < res.size() ? static_cast<int32_t>(res[j].second) : -1;
    }
  }
  return out;
}

}  // namespace ann
