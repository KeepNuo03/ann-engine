#include <algorithm>
#include <random>
#include <vector>

#include "ann/bruteforce.h"
#include "ann/distance.h"
#include "doctest.h"

namespace {
ann::Dataset random_dataset(size_t n, size_t d, unsigned seed) {
  ann::Dataset ds(n, d);
  std::mt19937 rng(seed);
  std::normal_distribution<float> g;
  for (size_t i = 0; i < n * d; ++i) ds.data()[i] = g(rng);
  return ds;
}
}  // namespace

TEST_CASE("single-query brute force returns sorted exact neighbors") {
  auto base = random_dataset(2000, 32, 1);
  auto q = random_dataset(1, 32, 2);
  auto res = ann::bruteforce_knn(base, q.row(0), 10);
  REQUIRE(res.size() == 10);
  CHECK(std::is_sorted(res.begin(), res.end()));

  // The 10th distance must be <= every distance not in the result.
  std::vector<bool> in(base.n(), false);
  for (auto& p : res) in[p.second] = true;
  for (size_t i = 0; i < base.n(); ++i)
    if (!in[i]) CHECK(ann::l2sq(q.row(0), base.row(i), 32) >= res.back().first);
}

TEST_CASE("a base vector is its own nearest neighbor") {
  auto base = random_dataset(1000, 16, 3);
  auto res = ann::bruteforce_knn(base, base.row(123), 1);
  CHECK(res[0].second == 123u);
  CHECK(res[0].first == 0.f);
}

TEST_CASE("batched brute force matches single-query version") {
  auto base = random_dataset(3000, 24, 4);
  auto qs = random_dataset(37, 24, 5);  // not a multiple of the query block
  auto batch = ann::bruteforce_knn_batch(base, qs, 10, 4);
  for (size_t i = 0; i < qs.n(); ++i) {
    auto single = ann::bruteforce_knn(base, qs.row(i), 10);
    for (size_t j = 0; j < 10; ++j) CHECK(batch.row(i)[j] == static_cast<int32_t>(single[j].second));
  }
}

TEST_CASE("k larger than n pads with -1") {
  auto base = random_dataset(5, 8, 6);
  auto qs = random_dataset(2, 8, 7);
  auto batch = ann::bruteforce_knn_batch(base, qs, 8, 1);
  CHECK(batch.row(0)[4] >= 0);
  CHECK(batch.row(0)[5] == -1);
}
