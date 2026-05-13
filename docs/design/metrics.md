# Metrics

> Status: design rationale for the metrics infrastructure under
> [`src/metrics/`](../../src/metrics), gated by the `metrics` Cargo
> feature. Companion to [`design.md`](design.md) §6.

cachekit's metrics surface is bigger than "two counters behind a
feature flag." It mirrors the cache trait hierarchy — recorder /
snapshot / exporter — so each concern lives in the smallest trait
that captures it, and policy code stays free of monitoring plumbing.
This document explains the three-trait separation, the
`&self`-vs-`&mut self` split, the `MetricsCell` interior-mutability
escape hatch, the Prometheus exporter contract, and what guarantees
counters do and do not provide.

## Goals and non-goals

The metrics module is shaped for:

- **Lightweight in-process counters** that a policy can increment on
  its hot path without measurable overhead when enabled.
- **Zero overhead when disabled.** The entire `metrics` module
  compiles away under `#[cfg(feature = "metrics")]`.
- **Decoupled consumption.** Tests, benchmarks, and production
  monitoring should each consume metrics in the shape they need
  without dragging recording concerns along.
- **Per-policy specificity.** A Clock policy's `hand_advance` count
  matters; a FIFO's `pop_oldest_empty_or_stale` count matters. The
  trait surface preserves these signals rather than flattening to
  one shape.

It is **not** shaped for:

- **High-cardinality labels.** Counters are flat scalars. Tag
  dimensions (per-key, per-tenant) are out of scope.
- **Histograms or sliding windows.** Counters and gauges only.
  Latency distributions live in the user's monitoring stack via
  external instrumentation.
- **Audit-grade accounting.** Counters use `Relaxed` atomics
  ([`src/store/weight.rs`](../../src/store/weight.rs)) and wrap on
  overflow in release. Best-effort observability, not financial
  ledger.

## Three-trait separation

```text
                                ┌─────────────────────────────┐
                                │     CoreMetricsRecorder     │
                                │  record_get_hit, _miss,     │
                                │  _insert_*, _evict_*,       │
                                │  _clear                     │
                                └──────────────┬──────────────┘
                                               │ extends
        ┌──────────┬───────────┬───────────────┼───────────┬────────────┐
        ▼          ▼           ▼               ▼           ▼            ▼
   FifoRec    LruRec       LfuRec          ArcRec      ClockRec    S3FifoRec
                │                                                       …
                ▼
            LruKRec
            (further extends LruRec)

   Consumption (decoupled from recording):
   ┌──────────────────────────────┐    ┌──────────────────────────────┐
   │ MetricsSnapshotProvider<S>   │    │ MetricsExporter<S>           │
   │ + MetricsReset               │    │ PrometheusTextExporter       │
   │ (bench / test)               │    │ (production monitoring)      │
   └──────────────────────────────┘    └──────────────────────────────┘
```

Three responsibilities, three trait families:

- **Record.** Per-policy `*MetricsRecorder` traits live in
  [`src/metrics/traits.rs`](../../src/metrics/traits.rs). Every
  policy-specific recorder extends `CoreMetricsRecorder` and adds
  policy-specific methods (`record_hand_advance` for Clock,
  `record_b1_ghost_hit` for ARC, etc.). The policy itself calls
  these methods on its hot path.
- **Snapshot.** `MetricsSnapshotProvider<S>` returns a `Copy`
  `*MetricsSnapshot` struct ([`src/metrics/snapshot.rs`](../../src/metrics/snapshot.rs))
  — a point-in-time scalar copy of every counter. Snapshots are
  `#[non_exhaustive]` for SemVer headroom and gated on `serde` for
  cross-process transport.
- **Export.** `MetricsExporter<S>` consumes a snapshot and pushes it
  to an external system. The shipped implementation,
  `PrometheusTextExporter` ([`src/metrics/exporter.rs`](../../src/metrics/exporter.rs)),
  writes Prometheus exposition format to any `W: Write + Send`.

Splitting these three lets:

- **Policy code stay minimal.** A policy needs only the recorder
  trait. It does not import snapshots or exporters.
- **Tests bypass production.** Bench harnesses use
  `MetricsSnapshotProvider` + `MetricsReset` and never touch
  `MetricsExporter`. Production code does the inverse.
- **Exporters multiply without policy churn.** Adding a StatsD or
  OpenTelemetry exporter is a new `impl MetricsExporter<S>` for the
  snapshot types — no policy changes.

## Per-policy recorder traits

Every policy gets its own recorder trait extending
`CoreMetricsRecorder`. The shipped set:

| Trait | Adds counters for |
|---|---|
| `FifoMetricsRecorder` | scan steps, stale skips, `pop_oldest` calls |
| `LruMetricsRecorder` | `pop_lru`, `peek_lru`, `touch`, `recency_rank` |
| `LruKMetricsRecorder` | extends `LruMetricsRecorder` + K-distance counters |
| `LfuMetricsRecorder` | `pop_lfu`, `peek_lfu`, frequency reads / mutates |
| `MfuMetricsRecorder` | mirrors LFU for most-frequent eviction |
| `ArcMetricsRecorder` | T1→T2 promotions, B1/B2 ghost hits, `p` movement |
| `CarMetricsRecorder` | recent→frequent, ghost hits, hand sweeps |
| `ClockMetricsRecorder` | hand advances, ref-bit resets |
| `ClockProMetricsRecorder` | cold↔hot transitions, test entries |
| `NruMetricsRecorder` | sweep steps, ref-bit resets |
| `SlruMetricsRecorder` | probationary→protected, protected evictions |
| `TwoQMetricsRecorder` | A1in→Am promotions, A1out ghost hits |
| `S3FifoMetricsRecorder` | promotions, main reinserts, ghost hits |

Two design principles drive the granularity:

- **Each counter answers a tuning question.** "Are my LRU-K
  promotions worth the metadata?" "Is my ARC ghost list catching
  meaningful hits?" Generic `evictions: u64` cannot answer either.
- **Counters live near their semantics.** `record_a1in_to_am_promotion`
  belongs to 2Q because A1in/Am are 2Q concepts. Putting it on
  `CoreMetricsRecorder` would force every other policy to either
  implement a meaningless method or document a no-op.

The trade is API surface: 14 recorder traits with ~5-10 methods
each. The mitigation is that **users do not implement them** — they
implement the shipped `*Metrics` structs through inherent methods on
each policy, and they read snapshots, not recorders.

## The `&self`-vs-`&mut self` split

Several `Cache<K, V>` methods take `&self`:
[`trait-hierarchy.md`](trait-hierarchy.md#peek-vs-get--the-readmutate-split)
explains why. The metrics system has to honour this — a `&self`
read path cannot call a `&mut self` recorder. The shipped solution
is a parallel `*MetricsReadRecorder` family for each policy whose
read paths increment counters:

| Mutable trait | Read-only counterpart |
|---|---|
| `FifoMetricsRecorder` | `FifoMetricsReadRecorder` |
| `LruMetricsRecorder` | `LruMetricsReadRecorder` |
| `LruKMetricsRecorder` | `LruKMetricsReadRecorder` |
| `LfuMetricsRecorder` | `LfuMetricsReadRecorder` |
| `MfuMetricsRecorder` | `MfuMetricsReadRecorder` |

The read-only traits take `&self` on every method. They are
implemented through interior mutability on the concrete metrics
struct — specifically `MetricsCell`, the internal type that wraps
`Cell<u64>` with an `unsafe impl Sync` (covered below).

Two questions this design avoided:

- **"Why not put `Cell<u64>` directly on the metrics struct?"**
  Because `Cell<u64>` is `!Sync`, which propagates and prevents
  every policy struct that embeds metrics from being `Sync`. The
  thin `MetricsCell` wrapper makes the synchronisation discipline
  explicit at one site instead of N.
- **"Why not just `AtomicU64` for everything?"** Because counters
  on `&mut self` paths (the majority — `insert`, `get`, `evict`)
  do not need atomic semantics; the policy already holds exclusive
  access. Using `AtomicU64` everywhere would impose memory-fence
  cost on the hot path for no concurrency benefit. The split
  reserves atomic-ish behaviour (`MetricsCell` + external lock) for
  read paths only.

## `MetricsCell`: interior mutability under external lock

```rust,ignore
#[repr(transparent)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct MetricsCell(Cell<u64>);

unsafe impl Sync for MetricsCell {}
unsafe impl Send for MetricsCell {}
```

This is the only `unsafe impl Sync` in the metrics surface. The
contract:

- **External synchronization is required.** `MetricsCell` lives
  inside a policy struct that is itself behind an `RwLock` in any
  concurrent wrapper (see [`concurrency.md`](concurrency.md)). The
  read lock serializes concurrent `&self` access; the cell is
  manipulated under that lock.
- **The cell is observation-only.** Lost increments are
  acceptable; the worst-case outcome is undercounting a metric,
  which is a precision issue, not a correctness one. Cache hits and
  evictions still behave correctly.
- **`pub(crate)`.** The type does not escape the crate.
  Down-stream code can read counters through the snapshot API but
  cannot construct `MetricsCell` itself, which prevents misuse from
  outside the codebase.

The alternatives considered and rejected:

- `Mutex<u64>` — cost dominates the counter increment.
- `AtomicU64` — works, but imposes fence cost where no concurrency
  exists for the increment itself.
- `RefCell<u64>` — runtime borrow checking with panic on contention;
  not desirable on a metrics increment path.

`MetricsCell` is the smallest tool that says "we know about the
sync requirement; trust the external lock; pay no per-increment
cost beyond a `Cell::set`."

## Snapshots: cheap, copyable, optionally serializable

Every snapshot struct in [`src/metrics/snapshot.rs`](../../src/metrics/snapshot.rs)
follows the same shape:

```rust,ignore
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LruMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub pop_lru_calls: u64,
    pub pop_lru_found: u64,
    pub peek_lru_calls: u64,
    pub peek_lru_found: u64,
    pub touch_calls: u64,
    pub touch_found: u64,
    pub recency_rank_calls: u64,
    pub recency_rank_found: u64,
    pub recency_rank_scan_steps: u64,
    pub cache_len: usize,
    pub insertion_order_len: usize,
    pub capacity: usize,
}
```

Five intentional properties:

- **`Copy`.** A snapshot is a flat block of `u64`s and `usize`s.
  Copying is a `memcpy` and snapshots can flow through channels,
  futures, and test assertions without ceremony.
- **`Default`.** Equivalent to "no operations recorded." Useful for
  test fixtures and explicit reset comparisons.
- **`#[non_exhaustive]`.** Adding a new counter (e.g. when a
  policy variant gains a new internal step) is a minor version
  bump. Downstream code matching on the struct must accept new
  fields gracefully — the standard `non_exhaustive` discipline.
- **`PartialEq + Eq`.** Snapshot equality is well-defined and
  useful in tests. Two snapshots compare equal iff every counter
  matches.
- **Optionally `serde`.** Gated on `serde`, not unconditional, so
  the metrics module doesn't drag serde into builds that don't
  want it.

Gauges (`cache_len`, `insertion_order_len`, `capacity`) live
alongside counters and snapshot together. The Prometheus exporter
writes the right `# TYPE` line for each, which matters for the
scraper.

## Recording is push, consumption is pull

Two operating models coexist:

- **Recording is push from the policy.** The policy calls
  `m.record_get_hit()` directly. The recorder method has the
  cheapest possible body (one `+= 1`). This is the hot-path
  contract.
- **Consumption is pull from the consumer.** Tests / benches /
  exporters call `m.snapshot()` whenever they want a value, and
  `MetricsReset::reset_metrics(&self)` when they want to clear.
  Nothing about the policy timing depends on consumption.

Specifically, the policy does **not** push to the exporter. There
is no observer-pattern hook from the recorder to the exporter, no
synchronous flush on every increment, and no async channel between
them. The pull model lets benches consume at known checkpoints
(once per iteration), and lets production scrapers poll on their
own cadence (every 10 s, every minute, etc.).

The cost of the pull model is that an exporter cannot react to a
specific event (e.g. "evictions spiked above N"). cachekit users
who need event-driven reactions instrument at the application
layer, not the metrics layer.

## Prometheus text exporter

The shipped exporter (`PrometheusTextExporter` in
[`src/metrics/exporter.rs`](../../src/metrics/exporter.rs)) writes
the Prometheus text exposition format to any `W: Write + Send`:

```rust,ignore
let exporter = PrometheusTextExporter::new("myapp_cache", io::stdout());
let snapshot = lru_cache.snapshot();
exporter.export(&snapshot);
```

Three design choices worth naming:

- **Per-prefix instance.** The prefix (`myapp_cache`) is set at
  construction, not per call. This keeps the call site simple and
  enforces a single metric namespace per exporter instance.
- **I/O errors are silently dropped.** A failing write does not
  panic the cache or surface a `Result`. The contract is
  "fire-and-forget monitoring" — a transient `EPIPE` to a metrics
  socket must not interrupt cache operations. Callers who need
  guaranteed delivery should wrap their writer in something with
  retry semantics and accept the cost.
- **The writer is `Mutex<W>`, not `RwLock<W>`.** Writing is
  always exclusive; there's no read path. Using `Mutex` here is
  the right primitive even though most of cachekit uses
  `parking_lot::RwLock`. (Note: this is `std::sync::Mutex`,
  poisoning-aware. `export` panics on poisoning. This is a
  deliberate divergence from `parking_lot` — the exporter is on
  the cold path and the std mutex's poisoning behaviour is fine
  there.)

Other exporters (StatsD, OpenTelemetry, custom) plug in by
implementing `MetricsExporter<S>` for each snapshot type they
care about. No changes elsewhere in the crate are required.

## Feature gating: all-or-nothing at compile time

The entire metrics subsystem is gated on the `metrics` Cargo
feature:

```rust
// src/lib.rs
#[cfg(feature = "metrics")]
pub mod metrics;
```

Inside each policy, recorder calls are wrapped:

```rust,ignore
#[cfg(feature = "metrics")]
self.metrics.record_get_hit();
```

When `metrics` is **off**:

- The entire `metrics` module disappears from the build.
- Every `record_*` call site becomes a no-op (the `#[cfg]` block
  compiles away).
- Snapshot types are not in the public API.
- Build time drops; binary size drops; no runtime cost.

When `metrics` is **on**:

- Recording costs one `u64 += 1` per call (or one `Cell::set` for
  read-only counters). For a 17-policy `DynCache` that records on
  every `get` / `insert`, the overhead is sub-nanosecond and shows
  up in benches as flat regression.
- The `metrics::snapshot` and `metrics::exporter` modules are in
  the public API and exporting infrastructure is available.

The trade-off is deliberate. No "low-cardinality always-on,
detailed-on-demand" two-tier scheme exists — every counter is
either always present (feature on) or absent (feature off). The
discipline that keeps "always present" cheap is the recorder
contract: methods do no work beyond incrementing a counter.

## What about `StoreMetrics`?

`StoreMetrics` ([`src/store/traits.rs`](../../src/store/traits.rs))
is a **separate**, simpler structure that ships unconditionally
(not behind `metrics`). It carries the universal counters every
store-layer implementation tracks:

```rust,ignore
pub struct StoreMetrics {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub updates: u64,
    pub removes: u64,
    pub evictions: u64,
}
```

The two systems coexist:

- `StoreMetrics` is the store-layer baseline. Always present, always
  cheap, six counters.
- `src/metrics/` (feature-gated) is the policy-layer detailed
  metrics — recorder traits, snapshots, exporter, per-policy signals.

A store typically backs `StoreMetrics` with `AtomicU64` counters
(see `StoreCounters` in [`src/store/weight.rs`](../../src/store/weight.rs)),
because stores are often behind concurrent wrappers and the
increment paths can be `&self`. The split mirrors the
sequential-vs-concurrent split at the trait level
([`concurrency.md`](concurrency.md)).

## Counter discipline

Three rules every recorder method follows:

1. **No allocation.** Counter increments are O(1) and allocation-free.
2. **No fallible operations.** A counter must not be in a position
   where it can fail — `+=` always succeeds; saturation is
   acceptable for u64 wrap (it takes years at billions/sec).
3. **No conditional logic beyond the counter itself.** A recorder
   method that branches on cache state belongs in the policy, not
   in metrics.

The corollary: a policy that wants a derived counter ("number of
evictions where the victim's recency rank was > 10") computes the
condition itself and calls one of two existing methods accordingly.
Putting the branching inside the recorder would couple metrics to
policy state.

## Adding a new metric

Checklist for adding a per-policy counter:

1. **Add the field.** Plain `u64` if it's updated on `&mut self`
   paths; `MetricsCell` if it's updated on `&self` paths. Place it
   in the corresponding `*Metrics` struct under
   [`src/metrics/metrics_impl.rs`](../../src/metrics/metrics_impl.rs).
2. **Add the recorder method.** On the relevant `*MetricsRecorder`
   trait (or its `*ReadRecorder` counterpart for `&self`).
3. **Implement on the policy's metrics struct.** One-line
   `+= 1` body.
4. **Wire the call site in the policy.** Wrap with
   `#[cfg(feature = "metrics")]`.
5. **Add the field to the snapshot.** In
   [`src/metrics/snapshot.rs`](../../src/metrics/snapshot.rs). The
   snapshot's `From<&*Metrics>` (or equivalent) needs the new
   field.
6. **Update the exporter.** Add a `write_counter` /
   `write_gauge` call in `PrometheusTextExporter::export` for the
   new field.

Six locations is a lot of friction for a new counter. The friction
is intentional — adding a counter is rarely the right answer to a
debugging question, and the friction encourages reuse of existing
counters where possible.

## Adding a new metric **type** (gauge vs counter, histogram)

Histograms and sliding windows are deliberately out of scope. Adding
either is a wider design change:

- The recorder traits assume `&mut u64 += 1` semantics. A histogram
  needs `observe(value)` semantics and an aggregation strategy.
- The snapshot types assume `Copy` and `u64` fields. A histogram
  snapshot needs bucket arrays.
- The Prometheus exporter writes counters and gauges only.

If histograms become needed (the most likely use case is latency
distribution per policy), the design has space: introduce a
`HistogramRecorder` trait alongside `CoreMetricsRecorder` and a
matching `HistogramSnapshot`. The existing exporter stays counter-
and-gauge-only; a new `PrometheusHistogramExporter` handles the
new shape. The current omission is a coverage decision, not a
foundation problem.

## Guarantees and non-guarantees

What the metrics system guarantees:

- **Eventual consistency in single-threaded builds.** Every recorded
  event eventually appears in `snapshot()` for the same thread.
- **Snapshot atomicity per counter.** A snapshot reads each
  counter as a single load; no torn `u64` reads on 64-bit
  platforms.
- **No cache correctness impact.** Metrics never block, panic
  (except `PrometheusTextExporter` on poisoned mutex), or alter
  cache state.

What it does **not** guarantee:

- **Cross-counter snapshot consistency.** A snapshot reads counters
  sequentially. A reader can observe `hits = 100, misses = 99`
  while a concurrent writer is mid-update; the next snapshot may
  show `hits = 100, misses = 101`. There is no "snapshot epoch."
- **Lossless recording under contention.** `MetricsCell`
  increments under a held read lock are safe; multiple read locks
  are not serialized against each other. Concurrent `&self`
  recorder calls on the same `MetricsCell` can lose increments.
  This is the "best-effort observability" caveat.
- **Wrap-safe arithmetic in release.** Release profile sets
  `overflow-checks = false`. Counters wrap silently. At one billion
  events per second, `u64` wraps in ~585 years — practically a
  non-issue, formally not a guarantee.

## See also

- [Design overview](design.md) — §6 frames metrics at the
  principles level
- [Cache trait hierarchy](trait-hierarchy.md) — `&self` / `&mut self`
  split that drives the read-vs-mutate recorder fork
- [Concurrency](concurrency.md) — read/write lock model behind
  `MetricsCell`'s soundness
- [Error model](error-model.md) — panic discipline shared by the
  exporter's poisoning behaviour
- [`src/metrics/`](../../src/metrics) — the canonical implementation
- [`src/store/traits.rs`](../../src/store/traits.rs) —
  `StoreMetrics`, the unconditional store-layer counterpart
