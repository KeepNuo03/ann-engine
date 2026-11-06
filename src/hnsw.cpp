#include "ann/hnsw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <random>
#include <stdexcept>

#include "ann/distance.h"
#include "ann/visited.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ann {

namespace {

using Neighbor = HNSWIndex::Neighbor;

struct Closer {  // min-heap on distance
  bool operator()(const Neighbor& a, const Neighbor& b) const { return a.first > b.first; }
};

thread_local VisitedList tl_visited;
thread_local size_t tl_dist_count = 0;

constexpr uint32_t kMagic = 0x57534e48;  // "HNSW"
constexpr uint32_t kVersion = 1;

}  // namespace

HNSWIndex::HNSWIndex(const Dataset& data, HNSWParams p)
    : data_(&data),
      n_(data.n()),
      d_(data.d()),
      M_(p.M),
      M0_(p.M0 ? p.M0 : 2 * p.M),
      efc_(p.ef_construction),
      seed_(p.seed),
      keep_pruned_(p.keep_pruned),
      mL_(1.0 / std::log(static_cast<double>(p.M))),
      level0_(n_ * (1 + M0_), 0),
      upper_(n_),
      levels_(n_, 0),
      node_locks_(new std::mutex[n_ ? n_ : 1]),
      global_lock_(new std::mutex) {
  if (M_ < 2) throw std::invalid_argument("M must be >= 2");
}

size_t HNSWIndex::last_distance_count() { return tl_dist_count; }

float HNSWIndex::dist(const float* a, uint32_t id) const {
  ++tl_dist_count;
  return l2sq(a, data_->row(id), d_);
}
float HNSWIndex::dist(uint32_t a, uint32_t b) const {
  return l2sq(data_->row(a), data_->row(b), d_);
}

uint32_t* HNSWIndex::links(uint32_t id, int level) {
  if (level == 0) return &level0_[static_cast<size_t>(id) * (1 + M0_)];
  return &upper_[id][static_cast<size_t>(level - 1) * (1 + M_)];
}
const uint32_t* HNSWIndex::links(uint32_t id, int level) const {
  return const_cast<HNSWIndex*>(this)->links(id, level);
}

std::vector<uint32_t> HNSWIndex::neighbors(uint32_t id, int level) const {
  const uint32_t* l = links(id, level);
  return std::vector<uint32_t>(l + 1, l + 1 + l[0]);
}

// Level of each node depends only on (seed, id), so serial and parallel
// builds assign the same levels.
int HNSWIndex::random_level(uint32_t id) const {
  std::mt19937_64 rng(seed_ * 0x9E3779B97F4A7C15ULL + id);
  // U in (0, 1]
  double u = (static_cast<double>(rng() >> 11) + 1.0) * (1.0 / 9007199254740992.0);
  int l = static_cast<int>(std::floor(-std::log(u) * mL_));
  return std::min(l, 254);
}

std::vector<Neighbor> HNSWIndex::search_layer(const float* q, const std::vector<Neighbor>& entries,
                                              size_t ef, int level, bool lock) const {
  VisitedList& visited = tl_visited;
  visited.resize(n_);
  visited.reset();

  std::priority_queue<Neighbor, std::vector<Neighbor>, Closer> candidates;  // closest on top
  std::priority_queue<Neighbor> results;                                    // farthest on top
  for (const auto& e : entries) {
    visited.mark(e.second);
    candidates.push(e);
    results.push(e);
  }
  while (results.size() > ef) results.pop();

  std::vector<uint32_t> buf;
  buf.reserve(std::max(M0_, M_));
  while (!candidates.empty()) {
    Neighbor c = candidates.top();
    if (c.first > results.top().first) break;  // nothing closer can be reached
    candidates.pop();

    const uint32_t* l;
    size_t cnt;
    if (lock) {
      std::lock_guard<std::mutex> g(node_locks_[c.second]);
      const uint32_t* src = links(c.second, level);
      buf.assign(src + 1, src + 1 + src[0]);
      l = buf.data();
      cnt = buf.size();
    } else {
      const uint32_t* src = links(c.second, level);
      l = src + 1;
      cnt = src[0];
    }
    for (size_t j = 0; j < cnt; ++j) {
      uint32_t e = l[j];
      if (!visited.try_mark(e)) continue;
      float de = dist(q, e);
      if (results.size() < ef || de < results.top().first) {
        candidates.emplace(de, e);
        results.emplace(de, e);
        if (results.size() > ef) results.pop();
      }
    }
  }
  std::vector<Neighbor> out;
  out.reserve(results.size());
  while (!results.empty()) {
    out.push_back(results.top());
    results.pop();
  }
  return out;  // farthest first
}

Neighbor HNSWIndex::greedy_descend(const float* q, Neighbor cur, int from_level, int stop_level,
                                   bool lock) const {
  std::vector<uint32_t> buf;
  for (int lv = from_level; lv > stop_level; --lv) {
    bool changed = true;
    while (changed) {
      changed = false;
      if (lock) {
        std::lock_guard<std::mutex> g(node_locks_[cur.second]);
        const uint32_t* src = links(cur.second, lv);
        buf.assign(src + 1, src + 1 + src[0]);
      } else {
        const uint32_t* src = links(cur.second, lv);
        buf.assign(src + 1, src + 1 + src[0]);
      }
      for (uint32_t e : buf) {
        float de = dist(q, e);
        if (de < cur.first) {
          cur = {de, e};
          changed = true;
        }
      }
    }
  }
  return cur;
}

std::vector<Neighbor> HNSWIndex::select_heuristic(const std::vector<Neighbor>& candidates,
                                                  size_t m) const {
  if (candidates.size() <= m) return candidates;
  std::vector<Neighbor> selected, pruned;
  selected.reserve(m);
  for (const auto& c : candidates) {
    if (selected.size() >= m) break;
    bool keep = true;
    for (const auto& s : selected)
      if (dist(c.second, s.second) < c.first) {  // closer to a selected neighbor than to q
        keep = false;
        break;
      }
    if (keep)
      selected.push_back(c);
    else
      pruned.push_back(c);
  }
  if (keep_pruned_)
    for (const auto& c : pruned) {
      if (selected.size() >= m) break;
      selected.push_back(c);
    }
  return selected;
}

void HNSWIndex::connect(uint32_t id, uint32_t new_id, float d_new, int level, bool lock) {
  std::unique_lock<std::mutex> g;
  if (lock) g = std::unique_lock<std::mutex>(node_locks_[id]);
  const size_t cap = level == 0 ? M0_ : M_;
  uint32_t* l = links(id, level);
  for (uint32_t j = 0; j < l[0]; ++j)
    if (l[1 + j] == new_id) return;
  if (l[0] < cap) {
    l[1 + l[0]] = new_id;
    ++l[0];
    return;
  }
  // Full: re-select among existing neighbors plus the new one.
  std::vector<Neighbor> cand;
  cand.reserve(cap + 1);
  cand.emplace_back(d_new, new_id);
  for (uint32_t j = 0; j < l[0]; ++j) cand.emplace_back(dist(id, l[1 + j]), l[1 + j]);
  std::sort(cand.begin(), cand.end());
  auto sel = select_heuristic(cand, cap);
  l[0] = static_cast<uint32_t>(sel.size());
  for (size_t j = 0; j < sel.size(); ++j) l[1 + j] = sel[j].second;
}

void HNSWIndex::insert(uint32_t id, bool parallel) {
  const int lvl = levels_[id];
  const float* q = data_->row(id);

  std::unique_lock<std::mutex> glock;
  if (parallel) glock = std::unique_lock<std::mutex>(*global_lock_);
  const int max_level = max_level_;
  const uint32_t entry = entry_;
  // Keep the global lock for the whole insert only when this node raises the
  // top level (rare); otherwise release it right away.
  if (parallel && lvl <= max_level) glock.unlock();

  if (max_level < 0) {  // first node
    entry_ = id;
    max_level_ = lvl;
    return;
  }

  Neighbor cur{dist(q, entry), entry};
  cur = greedy_descend(q, cur, max_level, lvl, parallel);

  std::vector<Neighbor> eps{cur};
  for (int lv = std::min(lvl, max_level); lv >= 0; --lv) {
    auto found = search_layer(q, eps, efc_, lv, parallel);  // farthest first
    std::reverse(found.begin(), found.end());                // closest first
    // A new node gets M links on every level (paper, and hnswlib); level 0 may
    // later grow up to M0 through back-links from newer nodes.
    auto sel = select_heuristic(found, M_);
    // In a parallel build another thread may already have linked itself to
    // this node on this level (it can reach us through an upper level before
    // our own level-lv list is written). Keep those edges instead of
    // overwriting them, or that thread's node can become unreachable.
    std::vector<uint32_t> early;
    {
      std::unique_lock<std::mutex> g;
      if (parallel) g = std::unique_lock<std::mutex>(node_locks_[id]);
      uint32_t* l = links(id, lv);
      early.assign(l + 1, l + 1 + l[0]);
      l[0] = static_cast<uint32_t>(sel.size());
      for (size_t j = 0; j < sel.size(); ++j) l[1 + j] = sel[j].second;
    }
    for (uint32_t e : early) connect(id, e, dist(id, e), lv, parallel);
    for (const auto& s : sel) connect(s.second, id, s.first, lv, parallel);
    eps = std::move(found);
  }

  if (lvl > max_level) {
    entry_ = id;
    max_level_ = lvl;
  }
}

void HNSWIndex::build(int threads) {
  if (n_ == 0) return;
  for (uint32_t i = 0; i < n_; ++i) {
    levels_[i] = static_cast<uint8_t>(random_level(i));
    if (levels_[i] > 0) upper_[i].assign(static_cast<size_t>(levels_[i]) * (1 + M_), 0);
  }
  insert(0, false);
  if (threads <= 1) {
    for (uint32_t i = 1; i < n_; ++i) insert(i, false);
  } else {
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
    for (long i = 1; i < static_cast<long>(n_); ++i) insert(static_cast<uint32_t>(i), true);
  }
}

std::vector<Neighbor> HNSWIndex::knn(const float* q, size_t k, size_t ef) const {
  tl_dist_count = 0;
  if (max_level_ < 0) return {};
  Neighbor cur{dist(q, entry_), entry_};
  cur = greedy_descend(q, cur, max_level_, 0, false);
  auto res = search_layer(q, {cur}, std::max(ef, k), 0, false);  // farthest first
  std::reverse(res.begin(), res.end());
  if (res.size() > k) res.resize(k);
  return res;
}

size_t HNSWIndex::knn_ids(const float* q, size_t k, size_t ef, uint32_t* out) const {
  auto res = knn(q, k, ef);
  for (size_t i = 0; i < res.size(); ++i) out[i] = res[i].second;
  return res.size();
}

size_t HNSWIndex::graph_bytes() const {
  size_t b = level0_.size() * sizeof(uint32_t) + levels_.size();
  for (const auto& u : upper_) b += u.size() * sizeof(uint32_t);
  return b;
}

std::string HNSWIndex::check_invariants() const {
  char msg[256];
  if (n_ == 0) return "";
  if (entry_ >= n_ || levels_[entry_] != max_level_) return "entry point is not at max level";
  for (uint32_t i = 0; i < n_; ++i) {
    for (int lv = 0; lv <= levels_[i]; ++lv) {
      const uint32_t* l = links(i, lv);
      size_t cap = lv == 0 ? M0_ : M_;
      if (l[0] > cap) {
        std::snprintf(msg, sizeof msg, "node %u level %d degree %u > %zu", i, lv, l[0], cap);
        return msg;
      }
      for (uint32_t a = 0; a < l[0]; ++a) {
        uint32_t e = l[1 + a];
        if (e >= n_) return "neighbor id out of range";
        if (e == i) {
          std::snprintf(msg, sizeof msg, "self loop at node %u level %d", i, lv);
          return msg;
        }
        if (levels_[e] < lv) {
          std::snprintf(msg, sizeof msg, "node %u links to %u on level %d it lacks", i, e, lv);
          return msg;
        }
        for (uint32_t b = a + 1; b < l[0]; ++b)
          if (l[1 + b] == e) {
            std::snprintf(msg, sizeof msg, "duplicate edge %u->%u level %d", i, e, lv);
            return msg;
          }
      }
    }
  }
  return "";
}

size_t HNSWIndex::unreachable_count() const {
  if (n_ == 0) return 0;
  std::vector<char> seen(n_, 0);
  std::vector<uint32_t> stack{entry_};
  seen[entry_] = 1;
  size_t count = 1;
  while (!stack.empty()) {
    uint32_t u = stack.back();
    stack.pop_back();
    const uint32_t* l = links(u, 0);
    for (uint32_t j = 0; j < l[0]; ++j)
      if (!seen[l[1 + j]]) {
        seen[l[1 + j]] = 1;
        ++count;
        stack.push_back(l[1 + j]);
      }
  }
  return n_ - count;
}

void HNSWIndex::save(const std::string& path) const {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open " + path);
  auto w64 = [&](uint64_t v) { std::fwrite(&v, 8, 1, f); };
  uint32_t hdr[2] = {kMagic, kVersion};
  std::fwrite(hdr, 4, 2, f);
  w64(n_);
  w64(d_);
  w64(M_);
  w64(M0_);
  w64(efc_);
  w64(seed_);
  int64_t ml = max_level_;
  std::fwrite(&ml, 8, 1, f);
  std::fwrite(&entry_, 4, 1, f);
  std::fwrite(levels_.data(), 1, n_, f);
  std::fwrite(level0_.data(), 4, level0_.size(), f);
  for (uint32_t i = 0; i < n_; ++i)
    if (levels_[i] > 0) std::fwrite(upper_[i].data(), 4, upper_[i].size(), f);
  if (std::fclose(f) != 0) throw std::runtime_error("write failed: " + path);
}

HNSWIndex HNSWIndex::load(const std::string& path, const Dataset& data) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  auto must = [&](size_t got, size_t want) {
    if (got != want) {
      std::fclose(f);
      throw std::runtime_error("truncated index file: " + path);
    }
  };
  uint32_t hdr[2];
  must(std::fread(hdr, 4, 2, f), 2);
  if (hdr[0] != kMagic || hdr[1] != kVersion) {
    std::fclose(f);
    throw std::runtime_error("not an ann-engine HNSW index: " + path);
  }
  uint64_t v[6];
  must(std::fread(v, 8, 6, f), 6);
  if (v[0] != data.n() || v[1] != data.d()) {
    std::fclose(f);
    throw std::runtime_error("index was built for a different dataset shape");
  }
  HNSWParams p;
  p.M = v[2];
  p.M0 = v[3];
  p.ef_construction = v[4];
  p.seed = v[5];
  HNSWIndex idx(data, p);
  int64_t ml;
  must(std::fread(&ml, 8, 1, f), 1);
  idx.max_level_ = static_cast<int>(ml);
  must(std::fread(&idx.entry_, 4, 1, f), 1);
  must(std::fread(idx.levels_.data(), 1, idx.n_, f), idx.n_);
  must(std::fread(idx.level0_.data(), 4, idx.level0_.size(), f), idx.level0_.size());
  for (uint32_t i = 0; i < idx.n_; ++i)
    if (idx.levels_[i] > 0) {
      idx.upper_[i].assign(static_cast<size_t>(idx.levels_[i]) * (1 + idx.M_), 0);
      must(std::fread(idx.upper_[i].data(), 4, idx.upper_[i].size(), f), idx.upper_[i].size());
    }
  std::fclose(f);
  return idx;
}

}  // namespace ann
