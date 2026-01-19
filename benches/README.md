# Benchmarks

Organized benchmark suite for cachekit cache policies.

## Structure

```
benches/
├── common/                    # Shared utilities
│   ├── mod.rs
│   ├── metrics.rs            # Benchmark metrics infrastructure
│   └── workload.rs           # Workload generators (zipfian, scan, etc.)
│
├── workloads.rs              # Cross-policy workload comparison (criterion)
├── ops.rs                    # Micro-operations (get_hit, insert latency)
├── comparison.rs             # External crate comparison (lru, quick_cache)
├── reports.rs                # Human-readable console reports (no criterion)
│
├── policy/                   # Policy-specific unique operations
│   ├── lru.rs                # pop_lru, touch
│   ├── lfu.rs                # pop_lfu, frequency_updates, bucket_touch
│   ├── lru_k.rs              # pop_lru_k, touch, k-value scaling
│   └── s3_fifo.rs            # scan_resistance, promotion
│
└── README.md
```

## Running Benchmarks

### All policies under identical workloads (recommended starting point)
```bash
cargo bench --bench workloads
```

### Micro-operations (get/insert latency per policy)
```bash
cargo bench --bench ops
```

### External crate comparison
```bash
cargo bench --bench comparison
```

### Policy-specific operations
```bash
cargo bench --bench policy_lru
cargo bench --bench policy_lfu
cargo bench --bench policy_lru_k
cargo bench --bench policy_s3_fifo
```

### Console reports (quick analysis without criterion overhead)
```bash
# Show available reports
cargo bench --bench reports

# Run specific report
cargo bench --bench reports -- hit_rate
cargo bench --bench reports -- scan
cargo bench --bench reports -- adaptation
cargo bench --bench reports -- detailed
cargo bench --bench reports -- memory
cargo bench --bench reports -- comprehensive

# Run all reports
cargo bench --bench reports -- all
```

## Benchmark Groups

### workloads.rs
Single source of truth for policy comparison. All policies tested under identical conditions:

| Group | Description |
|-------|-------------|
| `hit_rate` | Hit rate across workloads (uniform, hotset, scan, zipfian, etc.) |
| `scan_resistance` | Baseline → scan → recovery hit rate measurement |
| `adaptation_speed` | Workload shift response time |
| `comprehensive` | Full metrics: latency (p50/p95/p99), throughput, eviction stats |

### ops.rs
Per-operation latency (ns/op) for all policies:

| Group | Description |
|-------|-------------|
| `get_hit_ns` | Cache hit latency |
| `insert_evict_ns` | Insert with eviction latency |
| `mixed_ops_ns` | 80% hit / 20% miss workload |

### comparison.rs
Compare cachekit against external crates:

| Group | Description |
|-------|-------------|
| `get_hit` | Get performance vs lru, quick_cache |
| `insert_evict` | Insert performance vs lru, quick_cache |
| `mixed_workload` | Realistic workload comparison |
| `scaling_get` | Performance at different cache sizes |

### policy/*.rs
Operations unique to specific policies:

| File | Operations |
|------|------------|
| `policy_lru` | `pop_lru`, `touch_hotset` |
| `policy_lfu` | `pop_lfu`, `increment_frequency`, `bucket_touch`, eviction scaling |
| `policy_lru_k` | `pop_lru_k`, `touch_hotset`, K-value comparison |
| `policy_s3_fifo` | `scan_resistance`, `small_to_main_promotion`, high churn |

### reports.rs
Human-readable console reports (no criterion overhead, instant results):

| Report | Description |
|--------|-------------|
| `hit_rate` | Hit rate comparison table across core workloads |
| `extended` | Extended hit rate with all workload patterns |
| `scan` | Scan resistance (baseline/scan/recovery hit rates) |
| `adaptation` | Adaptation speed comparison with hit rate curves |
| `detailed` | Single benchmark showing all metrics fields |
| `memory` | Memory overhead comparison across policies |
| `comprehensive` | Full PolicyComparison tables for each policy |
| `all` | Run all reports sequentially |

## Latest Results

*Generated with `cargo bench --bench reports -- all` (capacity=4096, universe=16384, ops=200000)*

### Hit Rate Comparison

| Policy | uniform | hotset | scan | zipfian | scrambled | latest | scan_resist | flash_crowd |
|--------|---------|--------|------|---------|-----------|--------|-------------|-------------|
| LRU | 24.62% | 90.65% | 0.00% | 80.17% | 90.22% | **35.45%** | 16.75% | 88.50% |
| LRU-K | 24.61% | 90.64% | 0.00% | 82.57% | **91.13%** | 25.91% | 20.28% | 89.40% |
| LFU | 24.61% | 90.64% | 0.00% | 82.57% | **91.13%** | 25.82% | 20.28% | 89.40% |
| Heap-LFU | 24.67% | 90.67% | **22.52%** | 74.81% | 90.03% | 26.51% | 23.42% | 85.32% |
| Clock | 24.66% | 90.65% | 0.00% | 80.75% | 90.45% | 34.96% | 17.63% | 88.74% |
| S3-FIFO | 24.77% | 90.63% | 0.00% | **82.84%** | 91.07% | 23.45% | **24.00%** | **89.49%** |
| 2Q | 24.78% | 90.63% | 0.00% | 82.37% | 90.71% | 31.55% | 16.04% | 89.36% |

**Key insights**:
- **uniform**: All policies equal (~24.7%) - random access reaches theoretical limit
- **scan**: Heap-LFU alone survives (22.5%); all others collapse to 0%
- **zipfian**: S3-FIFO wins (82.8%) - frequency-aware policies outperform recency-only
- **latest**: LRU wins (35.5%) - recency-biased workload is LRU's sweet spot
- **scan_resistance**: S3-FIFO leads (24.0%) - handles mixed scan + point queries best

### Scan Resistance

| Policy | Baseline | During Scan | Recovery | Score |
|--------|----------|-------------|----------|-------|
| LRU | 79.65% | 7.03% | 68.54% | 0.86 |
| LRU-K | 79.66% | 7.69% | 78.54% | **0.99** |
| LFU | 79.66% | 7.69% | 78.54% | **0.99** |
| Heap-LFU | 79.21% | 21.89% | 75.79% | 0.96 |
| S3-FIFO | 79.66% | 7.69% | 78.82% | **0.99** |
| 2Q | 79.66% | 7.69% | 78.54% | **0.99** |
| Clock | 79.66% | 6.90% | 68.54% | 0.86 |

**Score** = recovery / baseline (1.0 = full recovery after scan pollution)

- **Winners (0.99)**: LRU-K, LFU, S3-FIFO, 2Q - recover almost fully after scans
- **Losers (0.86)**: LRU, Clock - scans permanently degrade performance by ~11%

### Adaptation Speed

| Policy | Ops to 50% | Ops to 80% | Stable HR |
|--------|------------|------------|-----------|
| LRU | 3,072 | 5,120 | **49.32%** |
| LRU-K | **1,024** | **2,048** | 9.08% |
| LFU | **1,024** | **2,048** | 9.08% |
| Heap-LFU | 2,048 | 2,048 | 9.86% |
| S3-FIFO | 8,192 | 11,264 | 44.34% |
| 2Q | 3,072 | 11,264 | 33.50% |
| Clock | 3,072 | 6,144 | **49.32%** |

- **Fastest warmup**: LRU-K reaches 80% of stable in only 2,048 ops
- **Slowest warmup**: S3-FIFO takes 11,264 ops (frequency tracking needs history)
- **Trade-off**: S3-FIFO has better long-term hit rates but slower warmup

### Throughput (ops/sec)

*Zipfian 1.0 workload, p99 latency in parentheses*

| Policy | uniform | zipfian | hotset | scan | loop_small |
|--------|---------|---------|--------|------|------------|
| LRU | 20.6M (125ns) | 16.2M (125ns) | 7.3M (375ns) | 4.6M (334ns) | 24.6M (42ns) |
| S3-FIFO | 19.9M (167ns) | 15.0M (167ns) | 24.3M (125ns) | 23.3M (84ns) | 26.2M (42ns) |
| Clock | **32.2M (84ns)** | **16.9M (84ns)** | **26.1M (83ns)** | **37.3M (42ns)** | 25.8M (42ns) |

- **Clock is fastest** with lowest tail latency across all workloads
- **S3-FIFO** trades ~12% throughput for better hit rates
- **LRU** struggles on scan workload (constant eviction overhead)

### Memory Overhead

*Shallow struct size (heap allocations not included)*

| Policy | Struct Size |
|--------|-------------|
| LRU | 56 bytes |
| Clock | 72 bytes |
| 2Q | 96 bytes |
| LRU-K | 104 bytes |
| S3-FIFO | 224 bytes |

S3-FIFO has 4x the metadata overhead of LRU due to its three-queue structure.

### Policy Selection Guide

| Use Case | Best Policy | Why |
|----------|-------------|-----|
| General purpose | **S3-FIFO** | Best hit rate on realistic workloads |
| Low latency critical | **Clock** | Fastest ops, lowest p99 |
| Scan-heavy / DB buffers | **S3-FIFO** or **2Q** | Excellent scan resistance |
| Fast warmup needed | **LRU** | Quickest to steady state |
| Memory constrained | **LRU** | Smallest metadata footprint |
| Mixed read/write | **LRU-K** | Good balance across patterns |

## Adding a New Policy

1. Add to `workloads.rs` - include in all benchmark groups
2. Add to `ops.rs` - include in get_hit and insert_evict benchmarks
3. (Optional) Create `policy/<name>.rs` if the policy has unique operations
4. Add `[[bench]]` entry to `Cargo.toml` if creating policy-specific file
