// Visited set for graph search, cleared in O(1) per query.
//
// Instead of a hash set, each node has a 16-bit tag. A query bumps the epoch
// and marks node i visited by writing tags[i] = epoch. When the epoch wraps
// around to 0, the whole array is zeroed once (every 65535 queries).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ann {

class VisitedList {
 public:
  explicit VisitedList(size_t n = 0) : tags_(n, 0) {}

  void resize(size_t n) {
    if (tags_.size() != n) {
      tags_.assign(n, 0);
      epoch_ = 0;
    }
  }

  // Starts a new query: every node becomes unvisited.
  void reset() {
    if (++epoch_ == 0) {
      std::fill(tags_.begin(), tags_.end(), 0);
      epoch_ = 1;
    }
  }

  bool visited(uint32_t id) const { return tags_[id] == epoch_; }
  void mark(uint32_t id) { tags_[id] = epoch_; }

  // Marks id and returns true if it was not visited before.
  bool try_mark(uint32_t id) {
    if (tags_[id] == epoch_) return false;
    tags_[id] = epoch_;
    return true;
  }

  uint16_t epoch() const { return epoch_; }
  const uint16_t* data() const { return tags_.data(); }

 private:
  std::vector<uint16_t> tags_;
  uint16_t epoch_ = 0;
};

}  // namespace ann
