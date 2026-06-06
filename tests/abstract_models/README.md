# Abstract Policy Models

Reference semantics for eviction policies used as **test-side oracles**. Exact and mirror models predict residency, hit/miss, and victims from access traces; integration tests in [`policy_semantics/`](../policy_semantics/) **dual-run** those models against real implementations. Bounded-tier policies use **invariant-only** tests (no `PolicyModel` dual-run today).

This is not a public API. For the full harness design (WCET concepts reused, CI, debugging), see [Policy semantic testing](../../docs/testing/static-analysis.md).

## Purpose

Cache policies are easy to get subtly wrong: tie-break order, peek vs get promotion, stale queue entries, and adaptive victim selection all drift without exhaustive traces. Abstract models give a **deterministic specification** that proptest and Miri smoke tests can compare against every step.

```
access trace ──► PolicyModel::apply(op) ──► ModelStep
                      │                         │
                      │                         ▼
                 real policy              assert residency,
                 (LruCore, etc.)          victim, rank, hit/miss
```

## Directory layout

```
abstract_models/
├── mod.rs          # Op, HitMiss, PolicyModel, ModelStep, proptest strategies
├── driver.rs       # Shared assertion helpers (assert_peek_victim, assert_models_agree, …)
├── reference/      # Spec-derived models (transcribed from docs/testing/specs/)
│   ├── fifo.rs     # NaiveFifoModel
│   ├── heap_lfu.rs # NaiveHeapLfuModel
│   ├── mfu.rs      # NaiveMfuModel
│   ├── lru.rs      # NaiveLruModel (timestamp formulation)
│   ├── lifo.rs     # NaiveLifoModel
│   ├── lfu.rs      # NaiveLfuModel
│   ├── lru_k.rs    # NaiveLruKModel
│   └── mru.rs      # NaiveMruModel
├── exact/          # Deterministic victims and residency
│   ├── lru.rs      # LruOccupancyModel (LRU, Fast-LRU, TTL layer)
│   ├── fifo.rs     # FifoModel
│   ├── lfu.rs      # LfuModel
│   └── …           # One module per exact/mirror policy
└── bounded/        # Invariant-only oracles (feature-gated doc stubs)
    ├── mod.rs      # re-exports bounded policy stubs
    ├── arc.rs      # doc stub — tests in policy_semantics/arc_tests.rs
    ├── car.rs      # doc stub
    ├── clock_pro.rs
    └── s3_fifo.rs
```

The harness is compiled into two integration test crates via `#[path]`:

- [`policy_semantics/main.rs`](../policy_semantics/main.rs) — proptest + smoke matrix
- [`ttl_integration_test.rs`](../ttl_integration_test.rs) — `LruOccupancyModel` + TTL deadlines

## Core types (`mod.rs`)

| Type | Role |
|------|------|
| `Op<K>` | Trace alphabet: `Insert`, `Get`, `Peek`, `GetMut`, `Touch`, `Remove`, `EvictOne` |
| `HitMiss` | `MustHit`, `MustMiss`, `MayHitOrMiss` (bounded / TTL only) |
| `PolicyModel<K>` | `apply(op) -> ModelStep`, `peek_victim_key`, `resident_set` |
| `ModelStep<K>` | `resident`, `hit`, `victim`, `evicted_on_insert` after one op |
| `OracleExpectation<K>` | `Exact(k)`, `Legal(set)`, `None` |

**Peek vs get:** `Peek` must not change `recency_rank`; `Get`, `GetMut`, and `Touch` promote on LRU-family policies. See [trait hierarchy](../../docs/design/trait-hierarchy.md).

## Model tiers

### Reference (`reference/`)

Spec-first oracles transcribed from [operational specs](../../docs/testing/specs/). Independent formulation from `exact/` models (e.g. LRU timestamps vs deque). Cross-model tests in `policy_semantics/` assert `reference/` agrees with `exact/` on the same traces.

| Policy | Reference model | Spec | Cross-model signal |
|--------|-----------------|------|-------------------|
| FIFO | `NaiveFifoModel` | [fifo.md](../../docs/testing/specs/policies/exact/fifo.md) | Drift guard (low day-one — `FifoModel` is already spec-shaped) |
| LRU | `NaiveLruModel` | [lru.md](../../docs/testing/specs/policies/exact/lru.md) | High — deque vs timestamp independence |
| Fast-LRU | `NaiveLruModel` (shared) | [fast-lru.md](../../docs/testing/specs/policies/exact/fast-lru.md) | Same reference; `op_strategy_with_get_mut` cross-model |
| LIFO | `NaiveLifoModel` | [lifo.md](../../docs/testing/specs/policies/exact/lifo.md) | Drift guard — `Vec` stack vs `VecDeque` exact |
| LFU | `NaiveLfuModel` | [lfu.md](../../docs/testing/specs/policies/exact/lfu.md) | High — `first_seen` log vs `FrequencyBuckets` |
| MRU | `NaiveMruModel` | [mru.md](../../docs/testing/specs/policies/exact/mru.md) | Drift guard — `Vec` index-0 vs `VecDeque` exact |
| Heap-LFU | `NaiveHeapLfuModel` | [heap-lfu.md](../../docs/testing/specs/policies/exact/heap-lfu.md) | High — `HashMap` Ord-min vs `BinaryHeap` exact |
| MFU | `NaiveMfuModel` | [mfu.md](../../docs/testing/specs/policies/exact/mfu.md) | High — `last_seq` map vs `BinaryHeap` exact |
| LRU-K | `NaiveLruKModel` | [lru-k.md](../../docs/testing/specs/policies/exact/lru-k.md) | High — `Vec` segments vs `VecDeque` exact |

FIFO and LRU have [TLA+ pilots](../../docs/testing/specs/formal/fifo/Fifo.tla) ([`Lru.tla`](../../docs/testing/specs/formal/lru/Lru.tla)) — read [tla-guide.md](../../docs/testing/specs/tla-guide.md); run [`scripts/run-fifo-tlc.sh`](../../scripts/run-fifo-tlc.sh) or [`scripts/run-lru-tlc.sh`](../../scripts/run-lru-tlc.sh) (manual, not CI).

### Exact (`exact/`)

Residency, victim, and recency rank (where applicable) must match the implementation exactly.

Examples: `LruOccupancyModel`, `FifoModel`, `LfuModel`, `MruModel`.

### Mirror (`exact/`)

Full internal state transcribed from the implementation rather than a simplified abstract rule. Used when the policy's behavior is defined by its data structure.

Examples: `ClockModel` (wraps `ClockRing`), `TwoQModel` (`TwoQCore` queues), `SlruModel`, `NruModel`.

### Bounded (`bounded/`)

Adaptive or scan-resistant policies where the victim is not uniquely determined from residency alone. Models assert:

- `len <= capacity`
- residency after inserts
- `debug_validate_invariants` / `check_invariants` on the real cache
- `OracleExpectation::Legal` victim sets (future: full legal-set checks)

Examples: ARC, CAR, Clock-PRO, S3-FIFO. Sibling files (`arc.rs`, etc.) are **documentation stubs** only; real checks live in `policy_semantics/*_tests.rs`. Submodules are gated by matching `policy-*` features (same as `exact/`).

## Harness modes

| Mode | Tests | Helper |
|------|-------|--------|
| **DualRun** | `prop_*_matches_model` | [`assert_dual_run_step`](driver.rs) or [`assert_dual_run_step_no_victim`](driver.rs) |
| **CrossModel** | `prop_*_naive_matches_current` | [`assert_models_agree`](driver.rs) |
| **InvariantOnly** | `prop_*_invariants` | [`run_invariant_trace`](driver.rs) |

Metadata and contributor checklist: [`spec_harness.rs`](spec_harness.rs).

## Policy coverage

**Canonical index:** [matrix.md](../../docs/testing/specs/matrix.md) (spec maturity, tier, harness mode, op strategy, traits).

Onboard a new policy using [template.md](../../docs/testing/specs/template.md).

## Proptest strategies

| Strategy | Use when |
|----------|----------|
| `standard_op_list()` | Default dual-run policies (LRU, FIFO, LFU, Clock, …) |
| `standard_op_list_no_evict()` | 2Q, SLRU (no `EvictingCache`; includes `Remove`) |
| `standard_op_list_mfu_safe()` | MFU only (skips `Remove`/`EvictOne`; stale heap) |
| `op_strategy_with_get_mut()` | Fast-LRU, S3-FIFO proptests |
| `short_op_list_no_evict()` | NRU (O(n) eviction; shorter traces) |

Capacities: `standard_capacity()` → `1..=16`. Trace lengths: `standard_op_list()` → `0..120` ops. Heap-LFU uses `standard_op_list()` (heap rebuild handles staleness on insert/evict).

## Running tests

```bash
# Full policy matrix (all features)
cargo test --test policy_semantics --all-features

# Single policy
cargo test --test policy_semantics --all-features prop_lru_core_matches_model

# High case count (CI)
PROPTEST_CASES=1000 cargo test --test policy_semantics --all-features

# Miri smoke traces only
cargo miri test --test policy_semantics --all-features smoke_ -- --test-threads=1

# TTL layer (shared LruOccupancyModel)
cargo test --test ttl_integration_test --features ttl
```

Proptests use `#[cfg_attr(miri, ignore)]`; Miri runs hand-written `smoke_*` traces only.

## Adding a new model

**Spec-first flow (recommended for exact policies):**

1. Write operational spec in [`docs/testing/specs/policies/<tier>/`](../../docs/testing/specs/policies/README.md) (state, per-`Op` rules, tie-breaks).
2. Implement `reference/<policy>.rs` from the spec only (independent formulation).
3. Implement or align `exact/<policy>.rs`; cite the spec doc in `//!` header.
4. Add cross-model tests: `prop_<policy>_naive_matches_current_model` using `assert_models_agree`.
5. Add impl dual-run: `run_ops`, `smoke_*`, `prop_*_matches_model`.
6. Gate features and append a row to [matrix.md](../../docs/testing/specs/matrix.md).

**Tier choice:** Simple deterministic eviction → `exact/`. Behavior tied to internal DS → mirror in `exact/`. Adaptive victim → `bounded/` (doc stub + invariant-only tests).

**Bounded policies:** add a `//!` doc stub in `bounded/<policy>.rs` (no `PolicyModel` required today); invariant-only `run_ops` in `policy_semantics/`.

Use `op_strategy_no_evict()` when the policy does not implement [`EvictingCache`](../../src/traits.rs).

### Dual-run pattern

All `policy_semantics/*_tests.rs` dual-run loops use shared helpers (except [`dual_impl_tests.rs`](../policy_semantics/dual_impl_tests.rs)).

Minimal (LIFO): [`lifo_tests.rs`](../policy_semantics/lifo_tests.rs)

```rust
assert_dual_run_step(cache, model, &step, |k| cache.contains(k), |_, _, _| {});
```

No `VictimInspectable` (MRU, mirror policies): use [`assert_dual_run_step_no_victim`](driver.rs) — see [`mru_tests.rs`](../policy_semantics/mru_tests.rs).

### Dual-run extra closure

Policy-specific oracles go in the `extra` closure:

- **LFU frequency:** [`lfu_tests.rs`](../policy_semantics/lfu_tests.rs)
- **LRU recency / peek:** [`lru_tests.rs`](../policy_semantics/lru_tests.rs), [`fast_lru_tests.rs`](../policy_semantics/fast_lru_tests.rs)
- **LRU-K access count:** [`lru_k_tests.rs`](../policy_semantics/lru_k_tests.rs)

```rust
assert_dual_run_step(cache, model, &step, |k| cache.contains(k), |cache, model, step| {
    for k in &step.resident {
        assert_eq!(cache.frequency(k), model.frequency(k));
    }
});
```

### Invariant-only pattern

Bounded tier — see [`arc_tests.rs`](../policy_semantics/arc_tests.rs):

```rust
run_invariant_trace(cache, ops, apply_arc_op, |cache| {
    cache.debug_validate_invariants();
});
```

For S3-FIFO, wrap `check_invariants` in `#[cfg(debug_assertions)]` inside the `check` closure.

Shared helpers live in [`driver.rs`](driver.rs): `assert_dual_run_step`, `assert_dual_run_step_no_victim`, `run_invariant_trace`, `assert_peek_victim`, `assert_recency_rank`, `assert_models_agree`, `assert_models_agree_with_recency`, and `probe_resident`.

## Debugging failures

1. Re-run the failing `smoke_*` test with `--nocapture`.
2. Shrink: `PROPTEST_CASES=1 cargo test --test policy_semantics prop_<policy> -- --nocapture`.
3. Step through `model.apply(op)` vs cache state after each op.
4. Check the adapter: `Arc<V>` vs `V`, `increment_frequency` vs `touch`, dereference `&Op` keys when matching.

## Related documentation

- [Policy spec matrix](../../docs/testing/specs/matrix.md)
- [Operational policy specs](../../docs/testing/specs/README.md)
- [Policy semantic testing (full harness)](../../docs/testing/static-analysis.md)
- [Testing strategy](../../docs/testing/testing.md)
- [Trait hierarchy](../../docs/design/trait-hierarchy.md)
- [Policy catalog](../../docs/policies/README.md)
