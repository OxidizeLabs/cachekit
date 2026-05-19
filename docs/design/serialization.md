# Serialization

> Status: design rationale for the current `serde` feature and the boundaries
> around future cache-state persistence. Companion to [`metrics.md`](metrics.md),
> [`ttl.md`](ttl.md), and [`builder-and-dyn-dispatch.md`](builder-and-dyn-dispatch.md).

cachekit has a narrow serialization surface today. The `serde` feature derives
`Serialize` / `Deserialize` for metrics snapshots and `StoreMetrics`; it does
**not** serialize cache contents, policy metadata, hash-map state, locks, or
builder dispatchers.

That boundary is intentional. Metrics are stable observations. Cache state is
live data with policy invariants, hash seeds, pointer-like handles, and optional
time semantics.

## Current Surface

With `features = ["serde"]`, these public value types derive serde:

- `StoreMetrics` in [`src/store/traits.rs`](../../src/store/traits.rs).
- Every metrics snapshot in [`src/metrics/snapshot.rs`](../../src/metrics/snapshot.rs).

Properties:

- They are flat value types (`u64`, `usize`, optional nested stats).
- They are `#[non_exhaustive]`, so new fields are SemVer-compatible at the Rust
  API level but still require schema discipline for serialized consumers.
- They carry observations, not live handles into cache internals.

No policy type implements serde today. No store type serializes entries today.

## Why Metrics Are Safe To Serialize

Metrics snapshots are point-in-time copies:

```rust,ignore
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LruMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    // ...
}
```

Serializing a snapshot cannot corrupt a cache on restore because there is no
restore into a running policy. At most, a downstream dashboard sees old or
partial counters. That matches the metrics contract: best-effort observability.

## Why Cache State Is Not Serialized

Serializing a cache is not just "serialize a map." A policy may contain:

- Intrusive list pointers or slot ids.
- Ghost-list history.
- Clock hand position and reference bits.
- ARC/CAR adaptive target parameters.
- Lazy heap tombstones.
- Hash seeds and randomized map order.
- `Arc<V>` sharing state.
- TTL deadlines based on monotonic time.

Restoring only keys and values discards policy warm state. Restoring every
internal field exposes private representation and risks accepting corrupted
state from disk.

The default position: **do not serialize policy internals until there is a
specific restore contract for that policy.**

## Two Possible Future Modes

If cache-state serialization lands later, it should choose one of two modes per
type.

### Data-only restore

Serialize only entries (`K`, `V`) plus capacity/config. On restore, rebuild the
policy as if entries were inserted in serialized order.

Pros:

- Simple and robust.
- No private invariants exposed.
- Cross-version friendly.

Cons:

- Loses recency/frequency/ghost history.
- Warm cache may behave cold after restore.
- Restore order becomes a semantic choice.

### Warm-state restore

Serialize policy metadata too: list order, frequency counters, clock hand,
ghost lists, ARC target, etc.

Pros:

- Better post-restore hit rate.
- Useful for long-lived caches that restart often.

Cons:

- Representation becomes part of the serialization contract.
- Every restore must validate invariants.
- Version migration becomes policy-specific.

Warm-state restore should be opt-in per policy, not a blanket derive.

## TTL and Time

TTL is the hardest serialization case because monotonic ticks are not portable
across process restarts. The TTL design doc recommends serializing **relative
remaining duration**, not raw `Instant`-derived ticks.

Rules for future TTL serialization:

- Never serialize raw monotonic `Tick` as if it were wall time.
- Capture remaining duration at serialization time.
- Restore by adding remaining duration to the new process clock.
- Expired-at-serialization entries should either be omitted or restored as
  expired and immediately purged. Prefer omission for data-only restore.
- Wall-clock deadlines require a separate API and explicit drift semantics.

This keeps `Clock` pluggable and avoids replaying meaningless old monotonic
values.

## Hash Seeds and Map Order

Do not serialize:

- `RandomState` seeds.
- `ShardSelector::randomized` key material.
- Hash-map bucket order.
- Internal `FxHashMap` iteration order.

Serialize semantic data only: keys, values, capacity, policy config, and, if
warm restore is explicitly chosen, policy metadata in a stable schema.

`ShardSelector::new(shards, seed)` is the exception because deterministic
routing is its public contract. If a type exposes deterministic sharding as
part of serialized config, the seed is config data and must be treated as
secret if keys are attacker-controlled.

## `Arc<V>` and Sharing

Several policies and stores use `Arc<V>`. Serialization should treat `Arc<V>`
as `V`, not as identity-preserving shared ownership:

- Do not attempt to preserve `Arc::ptr_eq` relationships.
- Do not serialize refcounts.
- Do not serialize weak references.

If multiple keys point at the same `Arc<V>`, data-only serialization will
duplicate the value unless the caller provides a higher-level interning scheme.
That is acceptable; cachekit should not infer value identity.

## Schema Discipline

For serialized artifacts controlled by cachekit (benchmark JSON, metrics
snapshots), use explicit schema rules:

- Additive optional fields are minor schema changes.
- Removing or renaming required fields is a major schema change.
- Stable identifiers should be constants, not string literals.
- Include enough metadata for interpretation: version, feature set where
  relevant, timestamp, and config.

For serde-derived Rust structs, `#[non_exhaustive]` is not enough for external
JSON compatibility. A downstream JSON consumer still sees fields. If stable
wire compatibility matters, introduce an explicit versioned artifact type
rather than serializing internal structs directly.

## What Not To Derive

Do not add `#[derive(Serialize, Deserialize)]` to a policy type just because it
compiles. Check:

- Does the serialized form expose private pointers, slot ids, or tombstones?
- Can deserialization validate every invariant?
- What happens if the target version has different metadata layout?
- Are hash seeds or time ticks being persisted accidentally?
- Does restoring this type produce a live, safe cache or only a bag of entries?

If the answer is not clear, add a separate DTO (`SerializableLruCache`) and a
fallible `try_from` restore path.

## See Also

- [Metrics](metrics.md) - current serde-supported snapshot types
- [TTL design](ttl.md) - relative TTL serialization recommendation
- [Hashing and key identity](hashing.md) - hash seeds and map order
- [Error model](error-model.md) - fallible restore should use `Result`
- [`src/metrics/snapshot.rs`](../../src/metrics/snapshot.rs)
- [`bench-support/src/json_results.rs`](../../bench-support/src/json_results.rs)
