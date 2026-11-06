// fvecs / ivecs readers. Vectors are stored row-major in one 64-byte aligned block.
//
// File format (TEXMEX): every record is a little-endian int32 dimension d,
// followed by d values (float32 for .fvecs, int32 for .ivecs).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ann {

struct AlignedFree {
  void operator()(void* p) const noexcept { std::free(p); }
};

// Owns n rows of d elements of T, contiguous and 64-byte aligned.
template <typename T>
class Matrix {
 public:
  Matrix() = default;
  Matrix(size_t n, size_t d) : n_(n), d_(d) {
    size_t bytes = n * d * sizeof(T);
    // aligned_alloc requires the size to be a multiple of the alignment.
    size_t rounded = (bytes + 63) / 64 * 64;
    if (rounded == 0) rounded = 64;
    void* p = std::aligned_alloc(64, rounded);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, rounded);
    data_.reset(static_cast<T*>(p));
  }

  size_t n() const { return n_; }
  size_t d() const { return d_; }
  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }
  T* row(size_t i) { return data_.get() + i * d_; }
  const T* row(size_t i) const { return data_.get() + i * d_; }
  size_t bytes() const { return n_ * d_ * sizeof(T); }

 private:
  size_t n_ = 0, d_ = 0;
  std::unique_ptr<T, AlignedFree> data_;
};

using Dataset = Matrix<float>;
using IdMatrix = Matrix<int32_t>;

namespace detail {

struct FileCloser {
  void operator()(std::FILE* f) const noexcept {
    if (f) std::fclose(f);
  }
};

// Reads a *vecs file. If max_rows > 0, stops after that many rows.
// If max_cols > 0, keeps only the first max_cols values of each row
// (used to keep top-k of a ground-truth file).
template <typename T>
Matrix<T> read_vecs(const std::string& path, size_t max_rows, size_t max_cols) {
  static_assert(sizeof(T) == 4, "vecs elements are 4 bytes");
  std::unique_ptr<std::FILE, FileCloser> f(std::fopen(path.c_str(), "rb"));
  if (!f) throw std::runtime_error("cannot open " + path);

  int32_t d32 = 0;
  if (std::fread(&d32, 4, 1, f.get()) != 1) throw std::runtime_error("empty file: " + path);
  if (d32 <= 0) throw std::runtime_error("bad dimension in " + path);
  const size_t d = static_cast<size_t>(d32);
  const size_t record = 4 + d * 4;

  std::fseek(f.get(), 0, SEEK_END);
  long size = std::ftell(f.get());
  if (size < 0 || static_cast<size_t>(size) % record != 0)
    throw std::runtime_error("file size is not a multiple of record size: " + path);
  size_t n = static_cast<size_t>(size) / record;
  if (max_rows > 0 && max_rows < n) n = max_rows;
  std::fseek(f.get(), 0, SEEK_SET);

  const size_t keep = (max_cols > 0 && max_cols < d) ? max_cols : d;
  Matrix<T> m(n, keep);
  std::unique_ptr<T[]> buf(new T[d]);
  for (size_t i = 0; i < n; ++i) {
    int32_t di = 0;
    if (std::fread(&di, 4, 1, f.get()) != 1) throw std::runtime_error("truncated file: " + path);
    if (static_cast<size_t>(di) != d)
      throw std::runtime_error("inconsistent dimension at row " + std::to_string(i) + " in " + path);
    if (std::fread(buf.get(), 4, d, f.get()) != d) throw std::runtime_error("truncated file: " + path);
    std::memcpy(m.row(i), buf.get(), keep * 4);
  }
  return m;
}

template <typename T>
void write_vecs(const std::string& path, const T* data, size_t n, size_t d) {
  std::unique_ptr<std::FILE, FileCloser> f(std::fopen(path.c_str(), "wb"));
  if (!f) throw std::runtime_error("cannot open for write " + path);
  int32_t d32 = static_cast<int32_t>(d);
  for (size_t i = 0; i < n; ++i) {
    std::fwrite(&d32, 4, 1, f.get());
    std::fwrite(data + i * d, 4, d, f.get());
  }
}

}  // namespace detail

inline Dataset read_fvecs(const std::string& path, size_t max_rows = 0) {
  return detail::read_vecs<float>(path, max_rows, 0);
}

// Ground truth: keep only the first k neighbor ids of each query (k = 0 keeps all).
inline IdMatrix read_ivecs(const std::string& path, size_t k = 0, size_t max_rows = 0) {
  return detail::read_vecs<int32_t>(path, max_rows, k);
}

inline void write_fvecs(const std::string& path, const float* data, size_t n, size_t d) {
  detail::write_vecs(path, data, n, d);
}

inline void write_ivecs(const std::string& path, const int32_t* data, size_t n, size_t d) {
  detail::write_vecs(path, data, n, d);
}

}  // namespace ann
