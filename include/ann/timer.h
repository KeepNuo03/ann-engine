#pragma once

#include <chrono>

namespace ann {

class Timer {
 public:
  Timer() : t0_(std::chrono::steady_clock::now()) {}
  void reset() { t0_ = std::chrono::steady_clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }
  double micros() const { return seconds() * 1e6; }

 private:
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace ann
