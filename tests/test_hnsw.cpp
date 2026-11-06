#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "ann/bruteforce.h"
#include "ann/hnsw.h"
#include "ann/visited.h"
#include "doctest.h"

namespace {

ann::Dataset random_dataset(size_t n, size_t d, unsigned seed) {
  ann::Dataset ds(n, d);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  for (size_t i = 0; i < n * d; ++i) ds.data()[i] = u(rng);
  return ds;
}

double recall10(const ann::HNSWIndex& idx, const ann::Dataset& base, const ann::Dataset& qs,
                size_t ef) {
  auto gt = ann::bruteforce_knn_batch(base, qs, 10, 4);
  double hit = 0;
  for (size_t i = 0; i < qs.n(); ++i) {
    auto res = idx.knn(qs.row(i), 10, ef);
    std::set<int32_t> truth(gt.row(i), gt.row(i) + 10);
    for (auto& r : res) hit += truth.count(static_cast<int32_t>(r.second));
  }
  return hit / (10.0 * static_cast<double>(qs.n()));
}

}  // namespace

TEST_CASE("visited list: epochs reset in O(1) and survive wraparound") {
  ann::VisitedList v(10);
  v.reset();
  CHECK(v.try_mark(3));
  CHECK_FALSE(v.try_mark(3));
  v.reset();
  CHECK_FALSE(v.visited(3));
  for (int i = 0; i < 70000; ++i) {  // crosses the 16-bit wrap
    v.reset();
    CHECK(v.epoch() != 0);
    if (i % 7 == 0) v.mark(static_cast<uint32_t>(i % 10));
  }
  v.reset();
  for (uint32_t i = 0; i < 10; ++i) CHECK_FALSE(v.visited(i));
}

TEST_CASE("HNSW on 10k random 32-d vectors reaches recall@10 >= 0.99 at ef=200") {
  auto base = random_dataset(10000, 32, 1);
  auto qs = random_dataset(200, 32, 2);
  ann::HNSWIndex idx(base, {});
  idx.build(1);
  CHECK(idx.check_invariants() == "");
  CHECK(idx.unreachable_count() == 0);
  double r = recall10(idx, base, qs, 200);
  MESSAGE("recall@10 at ef=200: " << r);
  CHECK(r >= 0.99);
  // recall must not get worse with a larger ef
  CHECK(recall10(idx, base, qs, 400) >= r - 1e-9);
}

TEST_CASE("results are sorted and a base vector finds itself") {
  auto base = random_dataset(3000, 16, 3);
  ann::HNSWIndex idx(base, {});
  idx.build(1);
  for (uint32_t i = 0; i < 3000; i += 97) {
    auto res = idx.knn(base.row(i), 10, 50);
    REQUIRE(res.size() == 10);
    CHECK(res[0].second == i);
    for (size_t j = 1; j < res.size(); ++j) CHECK(res[j - 1].first <= res[j].first);
  }
}

TEST_CASE("serial build is deterministic for a fixed seed") {
  auto base = random_dataset(4000, 16, 4);
  ann::HNSWIndex a(base, {}), b(base, {});
  a.build(1);
  b.build(1);
  REQUIRE(a.entry_point() == b.entry_point());
  REQUIRE(a.max_level() == b.max_level());
  for (uint32_t i = 0; i < 4000; ++i) {
    REQUIRE(a.level(i) == b.level(i));
    for (int lv = 0; lv <= a.level(i); ++lv) REQUIRE(a.neighbors(i, lv) == b.neighbors(i, lv));
  }
}

TEST_CASE("parallel build keeps invariants and recall") {
  auto base = random_dataset(10000, 32, 5);
  auto qs = random_dataset(200, 32, 6);
  ann::HNSWIndex idx(base, {});
  idx.build(4);
  CHECK(idx.check_invariants() == "");
  MESSAGE("unreachable after parallel build: " << idx.unreachable_count());
  CHECK(idx.unreachable_count() < base.n() / 100);
  double r = recall10(idx, base, qs, 200);
  MESSAGE("parallel recall@10 at ef=200: " << r);
  CHECK(r >= 0.98);
}

TEST_CASE("save and load give identical results") {
  auto base = random_dataset(3000, 16, 7);
  auto qs = random_dataset(50, 16, 8);
  ann::HNSWIndex idx(base, {});
  idx.build(1);
  const char* path = "/tmp/ann_test_hnsw.bin";
  idx.save(path);
  auto loaded = ann::HNSWIndex::load(path, base);
  CHECK(loaded.check_invariants() == "");
  CHECK(loaded.graph_bytes() == idx.graph_bytes());
  for (size_t i = 0; i < qs.n(); ++i) CHECK(idx.knn(qs.row(i), 10, 64) == loaded.knn(qs.row(i), 10, 64));
  std::remove(path);
}

TEST_CASE("tiny inputs") {
  auto base = random_dataset(1, 8, 9);
  ann::HNSWIndex idx(base, {});
  idx.build(1);
  auto res = idx.knn(base.row(0), 5, 10);
  REQUIRE(res.size() == 1);
  CHECK(res[0].second == 0u);
}
