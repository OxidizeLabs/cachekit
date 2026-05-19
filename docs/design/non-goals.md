# Non-Goals

> Status: explicit boundaries for cachekit's design. Companion to
> [`design.md`](design.md), which states what the crate optimizes for.

Good design needs negative space. This document records what cachekit is **not**
trying to be, so future features can be judged against the same boundaries.

## Not a Distributed Cache

cachekit is an in-process cache library. It does not provide:

- network protocols;
- replication;
- cluster membership;
- consistent hashing across processes;
- cross-node invalidation;
- persistence guarantees.

Use Redis, Memcached, or a database/cache service when those are the problem.
cachekit can still be useful inside a node in front of those systems.

## Not a Full Application Cache Framework

cachekit does not manage:

- request coalescing / singleflight;
- background refresh;
- cache stampede suppression;
- application-specific invalidation rules;
- loader functions or read-through APIs as the primary abstraction.

The library provides cache primitives and policies. Application frameworks can
compose them into higher-level behaviours.

## Not Async-Native Today

`AsyncCacheFuture` exists as a placeholder, but the shipped policies are
synchronous. Async-native traits are not currently implemented.

The reason is not that async is unimportant. It is that async cache APIs need
owned values, cancellation semantics, loader lifetime rules, and executor
integration. Adding `async fn get_or_insert_with` to the core trait would break
object safety and pull async choices into every policy.

Future async support should be a separate layer, not a mutation of
`Cache<K, V>`.

## Not `no_std`

cachekit uses:

- `std::collections`;
- `std::sync::Arc`;
- `std::time` in planned TTL work;
- `parking_lot` for concurrent wrappers;
- benchmark and metrics tooling built around `std`.

`no_std` would require a different allocator story, different synchronization
surface, and feature-gated alternatives for large parts of the crate. It is not
a current target.

## Not Lock-Free

The concurrency design is explicit and lock-based:

- `Concurrent*` wrappers use `parking_lot::RwLock`;
- sharded structures use one lock per shard;
- future lock-free reads are a research direction, not current design.

Lock-free structures would need a separate memory reclamation strategy,
different value ownership rules, and a much larger unsafe surface. The current
crate favours predictable, reviewable lock boundaries.

## Not a HashDoS Firewall

Some public surfaces use DoS-resistant hashing by default (`HashMapStore`,
`ClockRing`, `ShardSelector::randomized`). Other hot internal surfaces use
`FxHashMap` for speed.

cachekit documents those choices, but it is not a general-purpose security
boundary. Callers with adversarial keys must choose safe constructors, bound
admission, and avoid exposing interned handles or `total_weight` across trust
boundaries.

## Not a Serialization Format for Live Caches

The `serde` feature supports metrics snapshots and `StoreMetrics`, not live
cache state. Serializing a policy means deciding what to do with recency lists,
ghost history, clock hands, hash seeds, `Arc<V>` identity, and TTL deadlines.

Until a policy has an explicit restore contract, do not derive serde for it.

## Not a General Metrics Platform

The metrics layer provides counters, gauges, snapshots, reset, and a Prometheus
text exporter. It does not provide:

- high-cardinality labels;
- histograms;
- sampling;
- streaming events;
- tracing spans;
- alerting.

Use your monitoring stack for those. cachekit exposes enough counters to make
policy tuning possible without making the cache own observability.

## Not a Policy Research Playground at the Cost of Hot Paths

New policies are welcome, but they must fit the crate's constraints:

- no per-operation allocation in hot paths;
- predictable eviction cost;
- feature-gated implementation;
- docs and benchmarks;
- clear workload motivation.

A clever algorithm that needs tree walks, heap allocation on every access, or
opaque trait-object dispatch in the hot loop belongs in a research branch until
benchmarks justify it.

## Not a Replacement for Workload Analysis

cachekit ships many policies, but it cannot choose your workload for you.
`CachePolicy::Lru` or `CachePolicy::S3Fifo` are defaults, not guarantees. Users
still need to measure reuse distance, scan rate, write ratio, object sizes, and
tail latency under representative traffic.

The benchmark suite provides workload generators to help, but it cannot infer
production behaviour automatically.

## Not a Stability Promise for Internal Layout

Public traits and documented constructors follow SemVer. Internal layout does
not:

- slot ids;
- intrusive-list node fields;
- heap tombstone representation;
- ghost-list internals;
- metric recorder implementation details;
- `DynCache`'s private `CacheInner` enum.

If downstream code depends on private layout, it is outside the compatibility
contract.

## How To Use This Doc

When proposing a feature, ask:

1. Does it violate one of these non-goals?
2. If yes, is it a new layer that keeps the core intact?
3. Can it be feature-gated so users who do not need it pay nothing?
4. Does it preserve hot-path constraints?
5. Does it belong in cachekit, or in an application/framework crate above it?

If the answer is unclear, write a design doc before implementation.

## See Also

- [Design overview](design.md)
- [Concurrency](concurrency.md)
- [Serialization](serialization.md)
- [Metrics](metrics.md)
- [Benchmarking](benchmarking.md)
