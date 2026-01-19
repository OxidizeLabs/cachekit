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
├── workloads.rs              # Cross-policy workload comparison (single source of truth)
├── ops.rs                    # Micro-operations (get_hit, insert latency)
├── comparison.rs             # External crate comparison (lru, quick_cache)
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
cargo test --bench workloads -- --ignored --nocapture
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

## Latest Results

### Micro-ops (ns/op)

| Cache | get_hit | insert_evict |
|-------|---------|--------------|
| LRU | TBD | TBD |
| LRU-K | TBD | TBD |
| LFU | TBD | TBD |
| Clock | TBD | TBD |
| S3-FIFO | TBD | TBD |
| 2Q | TBD | TBD |

### Workload Throughput (Melem/s, 200k ops)

| Cache | uniform | hotset | scan | zipfian |
|-------|---------|--------|------|---------|
| LRU | TBD | TBD | TBD | TBD |
| LRU-K | TBD | TBD | TBD | TBD |
| LFU | TBD | TBD | TBD | TBD |
| Clock | TBD | TBD | TBD | TBD |
| S3-FIFO | TBD | TBD | TBD | TBD |
| 2Q | TBD | TBD | TBD | TBD |

## Adding a New Policy

1. Add to `workloads.rs` - include in all benchmark groups
2. Add to `ops.rs` - include in get_hit and insert_evict benchmarks
3. (Optional) Create `policy/<name>.rs` if the policy has unique operations
4. Add `[[bench]]` entry to `Cargo.toml` if creating policy-specific file
