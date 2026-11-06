# Results

Machine: 12th Gen Intel(R) Core(TM) i7-12700H, gcc 13.3.0, SIMD avx2

### Summary (single thread, smallest ef reaching the recall target)

| Recall target | Implementation | ef | Recall@10 | QPS | p99 (us) | Index build (s, all cores) |
|---|---|---|---|---|---|---|
| >= 0.95 | ours (v3) | 80 | 0.9750 | 9288 | 147 | 47.1 |
| >= 0.95 | ours (v0, naive visited set) | 80 | 0.9750 | 4312 | 358 | 47.1 |
| >= 0.95 | hnswlib | 80 | 0.9750 | 7572 | 249 | 56.0 |
| >= 0.99 | ours (v3) | 200 | 0.9955 | 4097 | 335 | 47.1 |
| >= 0.99 | ours (v0, naive visited set) | 200 | 0.9955 | 1887 | 773 | 47.1 |
| >= 0.99 | hnswlib | 200 | 0.9953 | 3522 | 468 | 56.0 |

### Ablation at ef=40 (single thread)

| Version | QPS | Recall@10 | p50 (us) | p99 (us) | vs previous | vs hnswlib |
|---|---|---|---|---|---|---|
| v0: unordered_set visited | 7695 | 0.9286 | 130 | 205 |  | 57% |
| v1: epoch visited list | 14066 | 0.9286 | 70 | 118 | +82.8% | 104% |
| v2: + interleaved level-0 layout | 13404 | 0.9286 | 74 | 137 | -4.7% | 99% |
| v3: + software prefetch | 16527 | 0.9286 | 60 | 90 | +23.3% | 122% |
| hnswlib (reference) | 13525 | 0.9276 | 72 | 152 |  | 100% |

### Ablation at ef=80 (single thread)

| Version | QPS | Recall@10 | p50 (us) | p99 (us) | vs previous | vs hnswlib |
|---|---|---|---|---|---|---|
| v0: unordered_set visited | 4312 | 0.9750 | 235 | 358 |  | 57% |
| v1: epoch visited list | 7746 | 0.9750 | 130 | 204 | +79.6% | 102% |
| v2: + interleaved level-0 layout | 8142 | 0.9750 | 124 | 197 | +5.1% | 108% |
| v3: + software prefetch | 9288 | 0.9750 | 110 | 147 | +14.1% | 123% |
| hnswlib (reference) | 7572 | 0.9750 | 130 | 249 |  | 100% |

### Ablation at ef=120 (single thread)

| Version | QPS | Recall@10 | p50 (us) | p99 (us) | vs previous | vs hnswlib |
|---|---|---|---|---|---|---|
| v0: unordered_set visited | 2873 | 0.9879 | 346 | 572 |  | 52% |
| v1: epoch visited list | 5519 | 0.9879 | 183 | 307 | +92.1% | 100% |
| v2: + interleaved level-0 layout | 5832 | 0.9879 | 176 | 241 | +5.7% | 106% |
| v3: + software prefetch | 6494 | 0.9879 | 158 | 220 | +11.3% | 118% |
| hnswlib (reference) | 5515 | 0.9882 | 183 | 303 |  | 100% |

### Ablation at ef=200 (single thread)

| Version | QPS | Recall@10 | p50 (us) | p99 (us) | vs previous | vs hnswlib |
|---|---|---|---|---|---|---|
| v0: unordered_set visited | 1887 | 0.9955 | 542 | 773 |  | 54% |
| v1: epoch visited list | 3375 | 0.9955 | 298 | 500 | +78.8% | 96% |
| v2: + interleaved level-0 layout | 3721 | 0.9955 | 278 | 378 | +10.2% | 106% |
| v3: + software prefetch | 4097 | 0.9955 | 251 | 335 | +10.1% | 116% |
| hnswlib (reference) | 3522 | 0.9953 | 288 | 468 |  | 100% |
