Designing high-performance caches in Rust is a multi-disciplinary problem: data structures, memory layout, concurrency, workload modeling, and systems-level performance all matter. The points below reflect what moves the needle in practice across systems, services, and libraries.

For interface and API decisions, the [Rust API Guidelines checklist](https://rust-lang.github.io/api-guidelines/checklist.html) is a useful companion for consistent, ergonomic design.

## 1. Workload First, Policy Second

Cache policy only matters relative to workload.

Identify access patterns:
- Hotset-heavy traffic: skewed keys, high churn.
- Scan-heavy traffic: large working sets, weak locality.
- Mixed traffic: bursts of hot data over large cold sets.

Measure:
- Reuse distance / stack distance.
- Read/write ratio.
- Temporal vs spatial locality.

Choose policies accordingly:
- LRU: good for temporal locality, bad for scans.
- LRU-K / 2Q (roadmap): better at filtering one-off accesses.
- Clock / ARC (roadmap): lower overhead, more adaptive.

Never design a "general purpose" cache first; design for the workload you expect.

## 2. Memory Layout Matters More Than Algorithms

In a cache, memory layout often dominates policy.

Prefer:
- Contiguous storage (Vec, slabs, arenas).
- Index-based indirection over pointer chasing.

Avoid:
- Excessive Box, Arc, linked lists.
- HashMap lookups in hot paths if avoidable.

Techniques:
- Store metadata (recency, freq, flags) in tightly packed structs.
- Separate hot metadata from cold payloads.
- Use slab allocators for fixed-size entries.

Cache misses caused by your own data structure are as bad as upstream misses.

## 3. Concurrency Strategy Is Core Design, Not a Wrapper

Locking strategy shapes everything.

Options:
- Global lock: simple, often fast enough for small cores, dies under high contention.
- Sharded caches: hash key -> shard, each shard independently locked.
- Lock-free or mostly-lock-free: hard in Rust, only worth it if contention dominates.

Rust-specific notes:
- When `std` is available, prefer `parking_lot` locks over `std::sync` for lower overhead and better ergonomics.
- Avoid Arc<Mutex<...>> in hot paths.
- Consider per-thread caches with periodic merge.
- Consider RCU-style read paths for read-heavy caches.

## 4. Avoid Per-Operation Allocation

Allocations kill throughput.

Pre-allocate:
- Entry pools.
- Node arrays.

Reuse:
- Free lists.
- Slabs.

Use:
- Vec with capacity management.
- Custom allocators if necessary.

Avoid:
- Creating new Arc, String, Vec per lookup.

If malloc shows up in your flamegraph, your cache is already slow.

## 5. Eviction Must Be Predictable and Cheap

Eviction is the critical slow path.

O(1) eviction is the goal.

Avoid unbounded tree walks or scans in eviction paths.

Maintain:
- Direct pointers/indices to eviction candidates.
- Eviction lists or clock hands.

Be careful with:
- Background eviction threads (synchronization overhead).
- Lazy cleanup that grows unbounded.

Eviction cost must be comparable to lookup cost, not orders of magnitude higher.

## 6. Metrics Are Not Optional

You cannot tune what you do not measure.

Track at least:
- Hit / miss rate.
- Eviction count and reason.
- Insert/update rate.
- Scan pollution rate.
- Lock contention or wait time (roadmap).

Expose:
- Lightweight counters in hot path.
- Optional detailed metrics behind feature flags.

Metrics should guide design decisions, not justify them afterward.

## 7. Separate Policy From Storage

Design in layers:
- Storage layer: how entries live in memory, allocation, layout, indexing.
- Policy layer: LRU, FIFO, LFU, LRU-K (roadmap: Clock/ARC/2Q, etc; see [Policy roadmap](policies/roadmap/README.md)); only manipulates metadata and ordering.
- Integration layer: ties application objects, payloads, or IDs into cache entries.

Related docs:
- [Policy overview](policies/README.md)
- [Policy roadmap](policies/roadmap/README.md)
- [Policy data structures](policy-ds/README.md)

This makes:
- Benchmarking easier.
- Policy experimentation cheap.
- Reasoning about performance clearer.

## 8. Beware of "Nice" Rust APIs in Hot Paths

Ergonomics often cost performance.

Avoid in critical loops:
- Heavy generics causing code bloat.
- Trait objects for hot dispatch.
- Closures capturing state.
- Iterator chains instead of simple loops.

Prefer:
- Explicit loops.
- Concrete types.
- Monomorphized fast paths.

You can wrap fast internals in nice APIs at the edges.

## 9. Scans Are the Enemy of Caches

In scan-heavy workloads:

Large sequential reads destroy LRU-style caches.

Solutions:
- Scan-resistant policies (LRU-K, 2Q/ARC are roadmap).
- Explicit "scan mode" hints from the caller or workload layer.
- Bypass cache for known one-shot reads.

If you ignore scans, your cache will look great in microbenchmarks and terrible in production.

## 10. Benchmark Like a System, Not a Library

Do not rely on random key benchmarks.

Use:
- Zipfian distributions.
- Mixed read/write workloads.
- Scan + point lookup mixtures.
- Time-varying hot sets.

Measure:
- Throughput.
- Tail latency.
- Memory overhead.
- Eviction cost.

A cache that is 5% faster on random keys but 50% worse under scans is a bad cache.

## 11. Rust-Specific Pitfalls

Arc is expensive in hot paths.

Borrow checker can push you toward indirection—fight it with:
- Index-based access.
- Interior mutability only where unavoidable.

Beware of:
- Hidden clones.
- Trait object dispatch.
- Over-generic designs.

Rust can be as fast as C, but only if you design like a systems programmer, not a library author.

## 12. Design for Failure Modes

Ask:
- What happens under memory pressure?
- What happens when eviction cannot keep up?
- What happens under pathological access patterns?

Add:
- Backpressure or rejection when full.
- Bypass modes.
- Emergency eviction strategies.

A cache that collapses under stress is worse than no cache.

## Bottom Line

High-performance caches are not about clever algorithms—they are about:
- Memory layout.
- Allocation discipline.
- Contention control.
- Eviction predictability.
- Workload realism.

In Rust, your main enemy is not safety—it is abstraction overhead and accidental allocation. Design from the metal upward, then wrap it in something pleasant to use.
