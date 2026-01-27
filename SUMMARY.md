# High-Performance Rust Cache Libraries (Summary)

This list focuses on Rust cache libraries that explicitly describe themselves as
high performance and/or publish benchmark results. It is not a definitive ranking,
but a curated summary based on each project's own documentation.

## Moka (crate: `moka`)
Moka is a fast, concurrent cache library inspired by Java's Caffeine. It provides
thread-safe sync and async caches with TinyLFU admission + LRU eviction, supports
entry count and weighted-size bounds, and includes TTL/TTI expiration policies.
Benchmark notes and hit-ratio results are published in its wiki.

Key features/benefits:
- High concurrency for reads and updates with thread-safe caches.
- Near-optimal hit ratio via TinyLFU admission and LRU eviction.
- Sync + async cache variants plus TTL/TTI and per-entry expiration.
- Documented benchmark results in the wiki.

Docs: https://github.com/moka-rs/moka
Benchmarks: https://github.com/moka-rs/moka/wiki#admission-and-eviction-policies

## Quick Cache (crate: `quick_cache`)
Quick Cache positions itself as a lightweight, high performance concurrent cache
optimized for low overhead. It uses the S3-FIFO policy, scales with threads, and
publishes benchmark comparisons in its README.

Key features/benefits:
- Low overhead relative to a concurrent hash table.
- S3-FIFO policy for scan resistance and strong hit rates.
- Good multi-thread scaling; async and sync atomic operations.
- Built-in benchmarks comparing to other caches.

Docs: https://github.com/arthurprs/quick-cache
Benchmarks: https://github.com/arthurprs/quick-cache#benchmarks

## Mini Moka (crate: `mini-moka`)
Mini Moka is a fast, concurrent cache library and a smaller, simpler sibling of
Moka. It offers thread-safe caches with high concurrency for reads/updates and a
non-thread-safe option for single-threaded use.

Key features/benefits:
- Fast, concurrent cache with simpler API and smaller footprint.
- Full concurrency of retrievals with high expected update concurrency.
- LFU admission and LRU eviction policies.

Docs: https://github.com/moka-rs/mini-moka

## foyer (crate: `foyer`)
foyer is an efficient hybrid cache library combining in-memory and disk-backed
cache. It emphasizes high concurrency, zero-copy in-memory caching, and pluggable
algorithms, aiming for performance under heavy load.

Key features/benefits:
- Hybrid in-memory + disk cache for larger working sets.
- Zero-copy in-memory abstraction for better performance.
- Concurrency-focused design with thread-safe primitives.
- Pluggable algorithms and built-in observability integrations.

Docs: https://github.com/foyer-rs/foyer
