// Benchmark driver: loads a dataset, runs one algorithm over a sweep of ef values
// and appends one CSV row per configuration.
//
//   ./bench --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
//           --gt data/sift/sift_groundtruth.ivecs --algo hnswlib
//           --M 16 --efc 200 --ef 10,20,40,80,120,200,400,800
//           --k 10 --threads 1 --index-cache results/hnswlib_M16_efc200.bin
//           --out results/hnswlib.csv
//
// --algo: brute | hnswlib   (hnsw is added in step 3)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ann/bruteforce.h"
#include "ann/dataset.h"
#include "ann/distance.h"
#include "ann/metrics.h"
#include "ann/timer.h"
#include "hnswlib/hnswlib.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Args {
  std::string base, query, gt, algo = "hnswlib", out, index_cache;
  size_t M = 16, efc = 200, k = 10, nq = 0;
  std::vector<size_t> ef = {10, 20, 40, 80, 120, 200, 400, 800};
  int threads = 1, build_threads = 0, repeats = 3;
};

std::vector<size_t> parse_list(const std::string& s) {
  std::vector<size_t> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stoul(tok));
  return v;
}

void usage() {
  std::fprintf(stderr,
               "usage: bench --base F --query F --gt F --algo brute|hnswlib [--M 16] [--efc 200]\n"
               "             [--ef 10,20,...] [--k 10] [--threads 1] [--build-threads N]\n"
               "             [--nq N] [--repeats 3] [--index-cache F] [--out results.csv]\n");
}

Args parse(int argc, char** argv) {
  Args a;
  std::map<std::string, std::string> kv;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
      usage();
      std::exit(2);
    }
    kv[key.substr(2)] = argv[++i];
  }
  auto get = [&](const char* k, std::string& dst) {
    if (kv.count(k)) dst = kv[k];
  };
  get("base", a.base);
  get("query", a.query);
  get("gt", a.gt);
  get("algo", a.algo);
  get("out", a.out);
  get("index-cache", a.index_cache);
  if (kv.count("M")) a.M = std::stoul(kv["M"]);
  if (kv.count("efc")) a.efc = std::stoul(kv["efc"]);
  if (kv.count("k")) a.k = std::stoul(kv["k"]);
  if (kv.count("nq")) a.nq = std::stoul(kv["nq"]);
  if (kv.count("ef")) a.ef = parse_list(kv["ef"]);
  if (kv.count("threads")) a.threads = std::stoi(kv["threads"]);
  if (kv.count("build-threads")) a.build_threads = std::stoi(kv["build-threads"]);
  if (kv.count("repeats")) a.repeats = std::stoi(kv["repeats"]);
  if (a.build_threads <= 0) a.build_threads = static_cast<int>(std::thread::hardware_concurrency());
  if (a.base.empty() || a.query.empty() || a.gt.empty()) {
    usage();
    std::exit(2);
  }
  return a;
}

std::string cpu_model() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("model name", 0) == 0) {
      auto p = line.find(':');
      std::string s = line.substr(p + 2);
      std::replace(s.begin(), s.end(), ',', ' ');
      return s;
    }
  return "unknown";
}

struct Row {
  std::string algo;
  size_t M = 0, efc = 0, ef = 0;
  int threads = 1;
  double recall = 0, qps = 0, p50 = 0, p99 = 0, build_s = 0;
  size_t graph_bytes = 0, vector_bytes = 0;
};

void write_rows(const std::string& path, const std::vector<Row>& rows) {
  if (path.empty()) return;
  bool exists = static_cast<bool>(std::ifstream(path));
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) throw std::runtime_error("cannot open " + path);
  if (!exists)
    std::fprintf(f,
                 "algo,M,efc,ef,threads,recall10,qps,p50_us,p99_us,build_s,graph_bytes,"
                 "vector_bytes,simd,cpu,compiler\n");
  const std::string cpu = cpu_model();
  for (const Row& r : rows)
    std::fprintf(f, "%s,%zu,%zu,%zu,%d,%.5f,%.1f,%.1f,%.1f,%.2f,%zu,%zu,%s,%s,%s\n",
                 r.algo.c_str(), r.M, r.efc, r.ef, r.threads, r.recall, r.qps, r.p50, r.p99,
                 r.build_s, r.graph_bytes, r.vector_bytes, ann::simd_name(), cpu.c_str(),
                 "gcc " __VERSION__);
  std::fclose(f);
}

// Runs `search(i, out_ids)` for every query. Single thread: timed per query for
// latency percentiles. Multi thread: only total wall time (QPS).
// Repeats the whole pass and keeps the median QPS run.
template <typename SearchFn>
void run_queries(size_t nq, size_t k, int threads, int repeats, SearchFn search,
                 std::vector<uint32_t>& ids, double& qps, double& p50, double& p99) {
  ids.assign(nq * k, 0);
  // warmup
  for (size_t i = 0; i < std::min<size_t>(nq, 200); ++i) search(i, &ids[i * k]);

  struct Run {
    double qps, p50, p99;
  };
  std::vector<Run> runs;
  for (int r = 0; r < repeats; ++r) {
    std::vector<double> lat;
    ann::Timer total;
    if (threads == 1) {
      lat.resize(nq);
      for (size_t i = 0; i < nq; ++i) {
        ann::Timer t;
        search(i, &ids[i * k]);
        lat[i] = t.micros();
      }
    } else {
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
      for (long i = 0; i < static_cast<long>(nq); ++i)
        search(static_cast<size_t>(i), &ids[static_cast<size_t>(i) * k]);
    }
    double secs = total.seconds();
    runs.push_back({static_cast<double>(nq) / secs, ann::percentile(lat, 50), ann::percentile(lat, 99)});
  }
  std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.qps < b.qps; });
  const Run& med = runs[runs.size() / 2];
  qps = med.qps;
  p50 = med.p50;
  p99 = med.p99;
}

// Ground truth restricted to the first nq queries.
ann::IdMatrix gt_prefix(const ann::IdMatrix& gt, size_t nq) {
  ann::IdMatrix out(nq, gt.d());
  std::memcpy(out.data(), gt.data(), nq * gt.d() * sizeof(int32_t));
  return out;
}

std::vector<Row> bench_brute(const Args& a, const ann::Dataset& base, const ann::Dataset& query,
                             const ann::IdMatrix& gt) {
  const size_t nq = gt.n(), k = a.k;
  // Recall from the batched (blocked, parallel) path, timing from the per-query path.
  ann::Timer t;
  ann::Dataset q(nq, query.d());
  std::memcpy(q.data(), query.data(), nq * query.d() * sizeof(float));
  ann::IdMatrix res = ann::bruteforce_knn_batch(base, q, k, a.threads);
  double batch_s = t.seconds();
  std::vector<uint32_t> ids(nq * k);
  for (size_t i = 0; i < nq * k; ++i) ids[i] = static_cast<uint32_t>(res.data()[i]);

  Row r;
  r.algo = a.threads == 1 ? "brute" : "brute_batch";
  r.threads = a.threads;
  r.recall = ann::mean_recall(ids, gt, k);
  r.qps = static_cast<double>(nq) / batch_s;
  r.vector_bytes = base.bytes();
  std::printf("%-12s threads=%d nq=%zu recall@%zu=%.4f qps=%.1f\n", r.algo.c_str(), a.threads, nq,
              k, r.recall, r.qps);
  return {r};
}

std::vector<Row> bench_hnswlib(const Args& a, const ann::Dataset& base,
                               const ann::Dataset& query, const ann::IdMatrix& gt) {
  const size_t n = base.n(), d = base.d(), nq = gt.n(), k = a.k;
  hnswlib::L2Space space(d);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  double build_s = 0;

  bool loaded = false;
  if (!a.index_cache.empty() && std::ifstream(a.index_cache)) {
    index = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, a.index_cache);
    loaded = true;
    std::printf("hnswlib: loaded index from %s\n", a.index_cache.c_str());
  } else {
    index = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, n, a.M, a.efc, 100);
    ann::Timer t;
    index->addPoint(base.row(0), 0);
#pragma omp parallel for schedule(dynamic, 256) num_threads(a.build_threads)
    for (long i = 1; i < static_cast<long>(n); ++i)
      index->addPoint(base.row(static_cast<size_t>(i)), static_cast<size_t>(i));
    build_s = t.seconds();
    std::printf("hnswlib: built M=%zu efc=%zu in %.1f s with %d threads\n", a.M, a.efc, build_s,
                a.build_threads);
    if (!a.index_cache.empty()) index->saveIndex(a.index_cache);
  }

  // Graph bytes: hnswlib stores level-0 links and vectors interleaved; report
  // level-0 link bytes plus upper-level link bytes separately from vectors.
  size_t l0_link_bytes = n * (index->size_links_level0_);
  size_t upper = 0;
  for (size_t i = 0; i < n; ++i)
    if (index->element_levels_[i] > 0) upper += index->size_links_per_element_ * index->element_levels_[i];

  std::vector<Row> rows;
  for (size_t ef : a.ef) {
    index->setEf(std::max(ef, k));
    auto search = [&](size_t i, uint32_t* out) {
      auto pq = index->searchKnn(query.row(i), k);
      size_t j = pq.size();
      while (!pq.empty()) {
        out[--j] = static_cast<uint32_t>(pq.top().second);
        pq.pop();
      }
    };
    std::vector<uint32_t> ids;
    Row r;
    run_queries(nq, k, a.threads, a.repeats, search, ids, r.qps, r.p50, r.p99);
    r.algo = "hnswlib";
    r.M = a.M;
    r.efc = a.efc;
    r.ef = ef;
    r.threads = a.threads;
    r.recall = ann::mean_recall(ids, gt, k);
    r.build_s = loaded ? 0 : build_s;
    r.graph_bytes = l0_link_bytes + upper;
    r.vector_bytes = n * d * sizeof(float);
    std::printf("hnswlib ef=%-4zu recall@%zu=%.4f qps=%9.1f p50=%7.1fus p99=%7.1fus\n", ef, k,
                r.recall, r.qps, r.p50, r.p99);
    rows.push_back(r);
  }
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    ann::Timer load;
    ann::Dataset base = ann::read_fvecs(a.base);
    ann::Dataset query = ann::read_fvecs(a.query);
    ann::IdMatrix gt_all = ann::read_ivecs(a.gt, a.k);
    size_t nq = a.nq > 0 ? std::min(a.nq, gt_all.n()) : gt_all.n();
    ann::IdMatrix gt = gt_prefix(gt_all, nq);
    std::printf("loaded base %zux%zu, %zu queries in %.2f s (simd=%s, cpu=%s)\n", base.n(),
                base.d(), nq, load.seconds(), ann::simd_name(), cpu_model().c_str());

    std::vector<Row> rows;
    if (a.algo == "brute")
      rows = bench_brute(a, base, query, gt);
    else if (a.algo == "hnswlib")
      rows = bench_hnswlib(a, base, query, gt);
    else
      throw std::runtime_error("unknown --algo " + a.algo);
    write_rows(a.out, rows);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
