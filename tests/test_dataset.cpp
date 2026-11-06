#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "ann/dataset.h"
#include "doctest.h"

namespace {

std::string tmp_path(const char* name) { return std::string("/tmp/ann_test_") + name; }

}  // namespace

TEST_CASE("fvecs round trip keeps every value and is 64-byte aligned") {
  const size_t n = 37, d = 130;  // d deliberately not a multiple of 8
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<float> src(n * d);
  for (auto& x : src) x = u(rng);

  auto path = tmp_path("roundtrip.fvecs");
  ann::write_fvecs(path, src.data(), n, d);
  ann::Dataset ds = ann::read_fvecs(path);

  REQUIRE(ds.n() == n);
  REQUIRE(ds.d() == d);
  CHECK(reinterpret_cast<uintptr_t>(ds.data()) % 64 == 0);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < d; ++j) CHECK(ds.row(i)[j] == src[i * d + j]);
  std::remove(path.c_str());
}

TEST_CASE("fvecs max_rows limits how many rows are read") {
  std::vector<float> src(10 * 4, 1.f);
  auto path = tmp_path("maxrows.fvecs");
  ann::write_fvecs(path, src.data(), 10, 4);
  CHECK(ann::read_fvecs(path, 3).n() == 3);
  std::remove(path.c_str());
}

TEST_CASE("ivecs reader keeps only the first k columns") {
  const size_t n = 5, d = 100;
  std::vector<int32_t> src(n * d);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<int32_t>(i);

  auto path = tmp_path("gt.ivecs");
  ann::write_ivecs(path, src.data(), n, d);
  ann::IdMatrix gt = ann::read_ivecs(path, 10);

  REQUIRE(gt.n() == n);
  REQUIRE(gt.d() == 10);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < 10; ++j) CHECK(gt.row(i)[j] == src[i * d + j]);
  std::remove(path.c_str());
}

TEST_CASE("inconsistent dimensions are rejected") {
  // Record 1 declares d=3, record 2 declares d=1 but still carries 3 floats, so the
  // total file size passes the size check and only the per-record check catches it.
  auto path = tmp_path("baddim.fvecs");
  std::FILE* f = std::fopen(path.c_str(), "wb");
  REQUIRE(f != nullptr);
  int32_t d3 = 3, d1 = 1;
  float v[3] = {1, 2, 3};
  std::fwrite(&d3, 4, 1, f);
  std::fwrite(v, 4, 3, f);
  std::fwrite(&d1, 4, 1, f);
  std::fwrite(v, 4, 3, f);
  std::fclose(f);

  CHECK_THROWS_AS(ann::read_fvecs(path), std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("truncated file is rejected") {
  auto path = tmp_path("trunc.fvecs");
  std::FILE* f = std::fopen(path.c_str(), "wb");
  REQUIRE(f != nullptr);
  int32_t d = 4;
  float v[2] = {1, 2};
  std::fwrite(&d, 4, 1, f);
  std::fwrite(v, 4, 2, f);  // only 2 of 4 floats
  std::fclose(f);

  CHECK_THROWS_AS(ann::read_fvecs(path), std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("missing file throws") {
  CHECK_THROWS_AS(ann::read_fvecs("/nonexistent/file.fvecs"), std::runtime_error);
}
