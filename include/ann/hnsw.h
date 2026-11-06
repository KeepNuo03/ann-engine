// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2018).
//
// Layout:
//  - Vectors are not copied; the index points into the caller's Dataset.
//  - Level 0 links: one flat uint32 array, node i owns (1 + M0) slots starting
//    at i * (1 + M0): slot 0 is the neighbor count, then up to M0 neighbor ids.
//  - Upper levels: only nodes with level > 0 get a vector holding, for each
//    level 1..L, (1 + M) slots in the same count-then-ids format.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ann/dataset.h"

namespace ann {

// Query-time implementation variants, used for the ablation table. All four
// run the same algorithm and return identical results; only speed differs.
enum class SearchVariant {
  kHashSet = 0,     // v0: std::unordered_set for visited nodes
  kEpoch = 1,       // v1: per-thread uint16 epoch tags (visited.h)
  kInterleaved = 2, // v2: + level-0 links and vector stored together per node
  kPrefetch = 3,    // v3: + software prefetch of the next neighbor's data
};
const char* variant_name(SearchVariant v);

struct HNSWParams {
  size_t M = 16;                // max neighbors on levels >= 1
  size_t M0 = 0;                // max neighbors on level 0 (0 means 2 * M)
  size_t ef_construction = 200;
  uint64_t seed = 100;
  bool keep_pruned = false;     // fill up to M with pruned candidates (paper's keepPrunedConnections)
};

class HNSWIndex {
 public:
  using Neighbor = std::pair<float, uint32_t>;  // (distance, id)

  HNSWIndex(const Dataset& data, HNSWParams params);

  // Inserts all vectors of the dataset. threads <= 1 builds serially and is
  // fully deterministic for a fixed seed.
  void build(int threads = 1);

  // k nearest neighbors of q, sorted by increasing distance.
  std::vector<Neighbor> knn(const float* q, size_t k, size_t ef) const;

  // Same, writing only ids into out[0..k). Returns how many were written.
  size_t knn_ids(const float* q, size_t k, size_t ef, uint32_t* out) const;

  // Chooses the query implementation. Interleaved variants build a packed
  // copy of level 0 (links + vector per node, 64-byte aligned) on first use.
  void set_variant(SearchVariant v);
  SearchVariant variant() const { return variant_; }
  size_t packed_bytes() const { return packed_stride_ * (packed_ ? n_ : 0); }

  void save(const std::string& path) const;
  // Loads a graph saved by save(); data must be the same dataset.
  static HNSWIndex load(const std::string& path, const Dataset& data);

  // Introspection for tests and reporting.
  size_t size() const { return n_; }
  size_t M() const { return M_; }
  size_t M0() const { return M0_; }
  int max_level() const { return max_level_; }
  uint32_t entry_point() const { return entry_; }
  int level(uint32_t id) const { return levels_[id]; }
  // Neighbors of id on a level (copy).
  std::vector<uint32_t> neighbors(uint32_t id, int level) const;
  size_t graph_bytes() const;
  // Returns an empty string if the graph is well formed, else a description of
  // the first problem found (self loop, duplicate, degree overflow, bad id,
  // link to a node that lacks that level, entry point not on the top level).
  std::string check_invariants() const;
  // Nodes not reachable from the entry point by following level-0 links.
  // Heuristic pruning can orphan a few nodes, more often in parallel builds
  // (hnswlib has the same property), so this is reported rather than asserted.
  size_t unreachable_count() const;
  // Distance computations performed by the last knn() call on this thread.
  static size_t last_distance_count();

 private:
  // Pointer to the count slot of id's adjacency list on a level.
  uint32_t* links(uint32_t id, int level);
  const uint32_t* links(uint32_t id, int level) const;

  int random_level(uint32_t id) const;
  void insert(uint32_t id, bool parallel);

  // Greedy beam search on one level from the given entry points.
  // Returns up to ef closest nodes as a max-heap-ordered vector (worst first).
  std::vector<Neighbor> search_layer(const float* q, const std::vector<Neighbor>& entries,
                                     size_t ef, int level, bool lock) const;
  // ef = 1 descent from top level down to (and excluding) stop_level.
  Neighbor greedy_descend(const float* q, Neighbor cur, int from_level, int stop_level,
                          bool lock) const;
  // Paper Algorithm 4 without extendCandidates. candidates sorted ascending.
  std::vector<Neighbor> select_heuristic(const std::vector<Neighbor>& candidates,
                                         size_t m) const;
  // Adds new_id to id's list on a level, pruning with the heuristic if full.
  void connect(uint32_t id, uint32_t new_id, float dist, int level, bool lock);

  template <class Visited, bool kInterleaved, bool kPrefetch>
  size_t search_level0(const float* q, Neighbor entry, size_t ef, size_t k, Neighbor* out) const;
  size_t query(const float* q, size_t k, size_t ef, Neighbor* out) const;
  void pack_level0();

  float dist(const float* a, uint32_t id) const;
  float dist(uint32_t a, uint32_t b) const;

  const Dataset* data_;
  size_t n_, d_, M_, M0_, efc_;
  uint64_t seed_;
  bool keep_pruned_;
  double mL_;

  std::vector<uint32_t> level0_;               // n * (1 + M0)
  std::vector<std::vector<uint32_t>> upper_;   // per node: level * (1 + M)
  std::vector<uint8_t> levels_;
  uint32_t entry_ = 0;
  int max_level_ = -1;

  SearchVariant variant_ = SearchVariant::kEpoch;  // set_variant() packs level 0 for v2/v3
  // Packed level 0: node i at packed_ + i * packed_stride_ holds
  // [count, id_0 .. id_{M0-1}, padding][vector (d floats)], 64-byte aligned.
  std::unique_ptr<char, AlignedFree> packed_;
  size_t packed_stride_ = 0, packed_vec_off_ = 0;

  std::unique_ptr<std::mutex[]> node_locks_;
  std::unique_ptr<std::mutex> global_lock_;
};

}  // namespace ann
