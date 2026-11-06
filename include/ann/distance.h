// Squared L2 distance. The SIMD path is chosen at compile time; the scalar
// version is always available as the reference implementation for tests.
#pragma once

#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ANN_SIMD_NAME "avx2"
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define ANN_SIMD_NAME "neon"
#else
#define ANN_SIMD_NAME "scalar"
#endif

namespace ann {

inline float l2sq_scalar(const float* a, const float* b, size_t d) {
  float acc = 0.f;
  for (size_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    acc += t * t;
  }
  return acc;
}

#if defined(__AVX2__) && defined(__FMA__)
// Sum the 8 lanes of a __m256.
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);                       // 4 lanes
  __m128 sh = _mm_movehdup_ps(lo);               // (1,1,3,3)
  __m128 s = _mm_add_ps(lo, sh);                 // (0+1, ., 2+3, .)
  sh = _mm_movehl_ps(sh, s);                     // (2+3, ...)
  s = _mm_add_ss(s, sh);
  return _mm_cvtss_f32(s);
}

// Two independent accumulators (16 floats per iteration) so consecutive FMAs
// do not wait on each other; the tail (< 8 floats) is finished in scalar code.
inline float l2sq_avx2(const float* a, const float* b, size_t d) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  for (; i + 8 <= d; i += 8) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
  }
  float acc = hsum256(_mm256_add_ps(acc0, acc1));
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    acc += t * t;
  }
  return acc;
}
#endif

#if defined(__ARM_NEON)
inline float l2sq_neon(const float* a, const float* b, size_t d) {
  float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
  size_t i = 0;
  for (; i + 8 <= d; i += 8) {
    float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
  }
  float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    acc += t * t;
  }
  return acc;
}
#endif

inline float l2sq(const float* a, const float* b, size_t d) {
#if defined(__AVX2__) && defined(__FMA__)
  return l2sq_avx2(a, b, d);
#elif defined(__ARM_NEON)
  return l2sq_neon(a, b, d);
#else
  return l2sq_scalar(a, b, d);
#endif
}

inline const char* simd_name() { return ANN_SIMD_NAME; }

}  // namespace ann
