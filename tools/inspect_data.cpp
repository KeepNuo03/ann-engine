// Loads SIFT1M and prints shapes, load time and a few sanity values.
// Usage: ./inspect_data [data_dir]   (default: data/sift)
#include <chrono>
#include <cstdio>
#include <string>

#include "ann/dataset.h"

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "data/sift";
  try {
    auto t0 = std::chrono::steady_clock::now();
    ann::Dataset base = ann::read_fvecs(dir + "/sift_base.fvecs");
    ann::Dataset query = ann::read_fvecs(dir + "/sift_query.fvecs");
    ann::IdMatrix gt = ann::read_ivecs(dir + "/sift_groundtruth.ivecs", 10);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("base:  n=%zu d=%zu (%.1f MB)\n", base.n(), base.d(), base.bytes() / 1e6);
    std::printf("query: n=%zu d=%zu\n", query.n(), query.d());
    std::printf("gt:    n=%zu k=%zu\n", gt.n(), gt.d());
    std::printf("load time: %.2f s\n", secs);
    std::printf("base[0][0..4]: %.0f %.0f %.0f %.0f\n", base.row(0)[0], base.row(0)[1],
                base.row(0)[2], base.row(0)[3]);
    std::printf("gt[0][0..4]:   %d %d %d %d\n", gt.row(0)[0], gt.row(0)[1], gt.row(0)[2],
                gt.row(0)[3]);

    bool ok = base.n() == 1000000 && base.d() == 128 && query.n() == 10000 &&
              query.d() == 128 && gt.n() == 10000 && gt.d() == 10;
    std::printf("%s\n", ok ? "OK: shapes match SIFT1M" : "UNEXPECTED SHAPES");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
