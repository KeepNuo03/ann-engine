#include <cmath>
#include <string>
#include <random>
#include <vector>

#include "ann/distance.h"
#include "doctest.h"

TEST_CASE("SIMD l2sq matches scalar for many dimensions") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-10.f, 10.f);
  for (size_t d : {1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 100, 128, 130, 960}) {
    std::vector<float> a(d), b(d);
    for (int rep = 0; rep < 20; ++rep) {
      for (size_t i = 0; i < d; ++i) {
        a[i] = u(rng);
        b[i] = u(rng);
      }
      double ref = 0;
      for (size_t i = 0; i < d; ++i) ref += double(a[i] - b[i]) * double(a[i] - b[i]);
      float s = ann::l2sq_scalar(a.data(), b.data(), d);
      float v = ann::l2sq(a.data(), b.data(), d);
      CAPTURE(d);
      CHECK(std::fabs(s - ref) / ref < 1e-5);
      CHECK(std::fabs(v - ref) / ref < 1e-5);
    }
  }
}

TEST_CASE("l2sq of identical vectors is zero") {
  std::vector<float> a(128, 3.5f);
  CHECK(ann::l2sq(a.data(), a.data(), 128) == 0.f);
}

TEST_CASE("simd path is reported") {
  MESSAGE("simd = " << std::string(ann::simd_name()));
  CHECK(ann::simd_name() != nullptr);
}
