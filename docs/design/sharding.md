# Sharding

> Status: design rationale for sharded data structures that exist today and
> roadmap notes for sharded cache policies. Companion to
> [`concurrency.md`](concurrency.md) and [`hashing.md`](hashing.md).

Sharding reduces contention by splitting one shared structure into N independent
substructures, each with its own lock and capacity accounting. cachekit already
uses this pattern at the data-structure and store layers. It does **not** yet
ship a generic `ShardedCache<C>` or sharded policy wrapper.

## Current Sharded Primitives

| Type | Layer | Purpose |
|---|---|---|
| `ShardedHashMapStore<K, V, S>` | store | N locked hash maps with global size counter |
| `ShardedSlotArena<T>` | data structure | N arenas addressed by `ShardedSlotId` |
| `ShardedFrequencyBuckets<K>` | data structure | N frequency bucket sets for concurrent LFU-style metadata |
| `ShardSelector` | helper | keyed hash routing from key to shard |

The sharded primitives are building blocks, not full cache policies. A future
`ShardedLruCache` would have to compose a sharded key index, per-shard recency
metadata, and global capacity semantics. That composition is where the hard
policy questions live.

## Why Shard?

A single `RwLock` wrapper is simple and often fast enough. It fails when:

- many threads mutate policy metadata (`get` on LRU, LFU, Clock);
- read paths still need atomics or lock acquisition;
- one hot lock dominates profile samples;
- cores spend more time waiting than doing cache work.

Sharding turns one contended lock into N less-contended locks. The cost is that
each shard is now a smaller cache with less global knowledge.

## Shard Routing

All routing should go through [`ShardSelector`](../../src/ds/shard.rs):

```rust,ignore
let selector = ShardSelector::randomized(16);
let shard = selector.shard_for_key(&key);
```

Routing requirements:

- Deterministic within a selector: same key maps to same shard.
- Uniform: no systematic bias toward lower shards.
- Keyed: adversaries should not be able to craft keys that all land on shard 0.
- Bounded: shard count is clamped to `[1, MAX_SHARDS]`.

Use `ShardSelector::randomized` unless reproducibility is required. If using
`ShardSelector::new(shards, seed)`, treat `seed` as secret when keys are
user-controlled.

## Capacity Semantics

Two capacity models are possible:

| Model | Behaviour | Pros | Cons |
|---|---|---|---|
| Per-shard capacity | total capacity split across shards | simple, one lock per op | hit rate fragmentation |
| Global capacity | one shared capacity budget | better utilization | cross-shard locking or global victim selection |

The primitives today mostly follow **per-shard local state with global gauges**:
each shard owns its data; aggregate `len` is tracked separately where needed.
This keeps operations single-lock. It also means a full shard can evict even if
another shard has spare room.

That is acceptable for stores and metadata primitives. For a full cache policy,
it is a hit-rate trade-off and must be documented at the policy level.

## Locking Discipline

Current sharded operations acquire at most **one shard lock**. This is the most
important invariant:

- No deadlock cycles.
- Lock hold time stays bounded by one shard operation.
- Callers do not need a global lock ordering table.

Any future operation that touches two shards must define an ordering rule, for
example "lock lower shard index first." Avoid two-shard operations unless the
hit-rate improvement justifies the concurrency risk.

## `ShardedSlotId`

`ShardedSlotArena<T>` cannot use a plain `SlotId`. A slot id must identify both
the shard and the local slot:

```text
ShardedSlotId = (shard_index, local_slot_id)
```

This is why sharding lives at the data-structure layer instead of being hidden
behind a generic wrapper. Once a policy stores handles, the handle type is part
of the policy's metadata layout.

## Global Metrics

Sharded types should expose aggregate metrics but record locally when possible.
The rule:

- Per-operation counters can be local or atomic.
- Gauges like total `len` need either an atomic aggregate or a shard scan.
- Snapshot consistency is best-effort; do not lock every shard just to make a
  metrics snapshot globally atomic.

This matches the metrics design: observability must not dominate the hot path.

## Roadmap: `ShardedCache<C>`

A generic sharded cache wrapper would look roughly like:

```rust,ignore
pub struct ShardedCache<C, K> {
    shards: Vec<RwLock<C>>,
    selector: ShardSelector,
    capacity_per_shard: usize,
    _key: PhantomData<K>,
}
```

Open questions:

- Does `C` have to be constructible by `CacheFactory`, or does the builder own
  all construction?
- Is capacity split evenly, weighted by shard traffic, or global?
- Do policies expose per-shard metrics only, or aggregate metrics too?
- How does `DynCache` integrate: `DynCache::Sharded(Box<...>)` or a sibling
  `DynShardedCache`?
- Should shard count be caller-specified, CPU-count-derived, or both?

The conservative first version should use per-shard capacity and one-lock
operations. Global victim selection should wait for benchmark evidence.

## When Not To Shard

- Cache fits on one lock without contention.
- Hit rate matters more than write throughput.
- Workload has a small hot set: all hot keys may still map to one shard.
- Cache capacity is small: per-shard fragmentation dominates.
- You need globally strict eviction order (true global LRU, ARC target `p`).

Sharding is a concurrency optimization, not a policy upgrade.

## See Also

- [Concurrency](concurrency.md)
- [Hashing and key identity](hashing.md)
- [Metrics](metrics.md)
- [`src/ds/shard.rs`](../../src/ds/shard.rs)
- [`src/store/hashmap.rs`](../../src/store/hashmap.rs)
- [`src/ds/slot_arena.rs`](../../src/ds/slot_arena.rs)
- [`src/ds/frequency_buckets.rs`](../../src/ds/frequency_buckets.rs)
