# Compatibility and Features

## MSRV

CacheKit targets the Rust MSRV listed in `Cargo.toml` and reflected in the README badge.

## Feature Flags

### General

- `metrics` — Enables hit/miss metrics and snapshots.
- `concurrency` — Enables concurrent wrappers (requires `parking_lot`).

### Eviction Policies (Per-Policy Feature Flags)

Each eviction policy can be enabled or disabled via its own feature flag. This keeps builds smaller when you only need specific policies.

| Feature | Policy | Description |
|---------|--------|-------------|
| `policy-fifo` | FIFO | First In, First Out |
| `policy-lru` | LRU | Least Recently Used (Arc-wrapped, `ConcurrentLruCache` available) |
| `policy-fast-lru` | Fast LRU | Optimized single-threaded LRU (~7–10× faster than LRU) |
| `policy-lru-k` | LRU-K | Scan-resistant with K-th access |
| `policy-lfu` | LFU | Least Frequently Used (bucket-based) |
| `policy-heap-lfu` | Heap LFU | LFU with heap-based eviction |
| `policy-two-q` | 2Q | Two-Queue |
| `policy-s3-fifo` | S3-FIFO | Scan-resistant three-queue FIFO |
| `policy-arc` | ARC | Adaptive Replacement Cache |
| `policy-lifo` | LIFO | Last In, First Out |
| `policy-mfu` | MFU | Most Frequently Used |
| `policy-mru` | MRU | Most Recently Used |
| `policy-random` | Random | Random eviction |
| `policy-slru` | SLRU | Segmented LRU |
| `policy-clock` | Clock | Second-chance clock |
| `policy-clock-pro` | Clock-PRO | Scan-resistant clock |
| `policy-nru` | NRU | Not Recently Used |

**Default features** include `policy-s3-fifo`, `policy-lru`, `policy-fast-lru`, `policy-lru-k`, and `policy-clock`.

**Convenience feature:** `policy-all` enables every policy above.

**Minimal builds:** Use `default-features = false` and select only the policies you need:

```toml
[dependencies]
cachekit = { version = "0.3", default-features = false, features = ["policy-lru", "policy-s3-fifo"] }
```

## Optional Dependencies

- `parking_lot` — Used for concurrent wrappers behind the `concurrency` feature.
