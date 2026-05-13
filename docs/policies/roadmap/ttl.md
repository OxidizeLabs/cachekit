# TTL / Time-Based Expiration

TTL is not a replacement policy; it's an expiration rule that coexists
with an eviction policy. The implementation status below tracks how
`cachekit` exposes it today.

## Status

Phase 1 (the "First Slice" from [`docs/design/ttl.md`](../../design/ttl.md))
is **landed** behind the `ttl` feature flag.

| Piece | Status | Location |
|---|---|---|
| `Clock` trait, `StdClock`, `MockClock` | landed | [`src/time.rs`](../../../src/time.rs) |
| `ExpirationIndex<K>` over `LazyMinHeap<K, u64>` | landed | [`src/ds/expiration_index.rs`](../../../src/ds/expiration_index.rs) |
| `LazyMinHeap::peek_best` primitive | landed | [`src/ds/lazy_heap.rs`](../../../src/ds/lazy_heap.rs) |
| `Tick`, `TtlStatus`, `ExpiringCache<K, V>` capability trait | landed | [`src/traits.rs`](../../../src/traits.rs) |
| `Expiring<C, K, V, T>` decorator | landed | [`src/policy/expiring.rs`](../../../src/policy/expiring.rs) |
| `impl Cache<K, V> for DynCache<K, V>` | landed | [`src/builder.rs`](../../../src/builder.rs) |
| `DynExpiringCache<K, V>` + `CacheBuilder::with_default_ttl` | landed | [`src/builder.rs`](../../../src/builder.rs) |
| `expirations` counter on `Expiring` (gated by `metrics`) | landed | same |
| Integration tests + proptest | landed | [`tests/ttl_integration_test.rs`](../../../tests/ttl_integration_test.rs) |
| Overhead bench (`ttl_overhead`) | landed | [`benches/ttl_overhead.rs`](../../../benches/ttl_overhead.rs) |

Phase 2 (deferred):

- Per-policy embedded `expires_at: u64` in `LruCore::Node` /
  `S3FifoCache::Node` via opt-in const generic or `LruWithTtl<K, V>` type.
  Gated on bench results from Phase 1.
- `ConcurrentExpiring<C>` with `Arc<V>` returns (no `Cache<K, V>` impl).
- Timer-wheel swap-in for `ExpirationIndex` as an alternative to the
  min-heap.
- `serde` support for `Tick` / `TtlStatus`.

## Quick Start

```rust,ignore
use cachekit::builder::{CacheBuilder, CachePolicy};
use std::time::Duration;

let mut cache = CacheBuilder::new(1024)
    .with_default_ttl(Duration::from_secs(60))
    .build::<u64, String>(CachePolicy::FastLru);

cache.insert(1, "value".to_string());
cache.insert_with_ttl(2, "fast".to_string(), Duration::from_millis(10));

// `peek` and `contains` hide expired entries logically; `get`, `insert`,
// `remove`, `purge_expired` physically purge them.
```

## Design Doc

The authoritative semantic contract — composition decision, ordering
invariant, clock semantics, overflow behaviour, and serialization notes —
lives in [`docs/design/ttl.md`](../../design/ttl.md).

## References

- Wikipedia: <https://en.wikipedia.org/wiki/Cache_replacement_policies>
