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
├── driver.rs       # Shared assertion helpers (assert_peek_victim, probe_resident, …)
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

## Policy coverage

| Policy | Model | Tier | Module |
|--------|-------|------|--------|
| LRU / Fast-LRU | `LruOccupancyModel` | exact | `exact/lru.rs` |
| FIFO | `FifoModel` | exact | `exact/fifo.rs` |
| LIFO | `LifoModel` | exact | `exact/lifo.rs` |
| MRU | `MruModel` | exact | `exact/mru.rs` |
| LFU | `LfuModel` | exact | `exact/lfu.rs` |
| Heap-LFU | `HeapLfuModel` | exact | `exact/heap_lfu.rs` |
| MFU | `MfuModel` | exact | `exact/mfu.rs` |
| LRU-K | `LruKModel` | exact | `exact/lru_k.rs` |
| Clock | `ClockModel` | mirror | `exact/clock.rs` |
| 2Q | `TwoQModel` | mirror | `exact/two_q.rs` |
| SLRU | `SlruModel` | mirror | `exact/slru.rs` |
| NRU | `NruModel` | mirror | `exact/nru.rs` |
| S3-FIFO | bounded checks | bounded | `bounded/s3_fifo.rs` |
| ARC | bounded checks | bounded | `bounded/arc.rs` |
| CAR | bounded checks | bounded | `bounded/car.rs` |
| Clock-PRO | bounded checks | bounded | `bounded/clock_pro.rs` |
| TTL | `LruOccupancyModel` + deadlines | composed | `ttl_integration_test.rs` |

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

1. **Choose a tier.** Simple deterministic eviction → `exact/`. Behavior tied to internal DS → mirror in `exact/`. Adaptive victim → `bounded/`.
2. **Add model code or doc stub:**
   - exact/mirror → implement `PolicyModel<K>` in `exact/<policy>.rs`
   - bounded → add a `//!` doc stub in `bounded/<policy>.rs` (no `PolicyModel` required today)
3. **Document tie-breaks** in the module `//!` doc (cite the implementation source, e.g. `LruCore` list order).
4. **Add tests** in `policy_semantics/<policy>_tests.rs`:
   - exact/mirror: `run_ops` dual-run adapter, `smoke_*`, `prop_*`
   - bounded: invariant-only `run_ops`, `smoke_*`, `prop_*` calling `debug_validate_invariants` / `check_invariants`
5. **Gate** the test module in `policy_semantics/main.rs` with `#[cfg(feature = "policy-…")]`.
6. **Update** the policy matrix in [static-analysis.md](../../docs/testing/static-analysis.md).

Use `op_strategy_no_evict()` when the policy does not implement [`EvictingCache`](../../src/traits.rs).

### Dual-run pattern

```rust
fn run_ops(cache: &mut LruCore<u8, u8>, model: &mut LruOccupancyModel<u8>, ops: &[Op<u8>]) {
    for op in ops {
        let step = model.apply(op.clone());
        // apply op to cache …
        assert_eq!(resident_set(cache), step.resident);
        assert_peek_victim(cache, model);
    }
}
```

Shared helpers live in [`driver.rs`](driver.rs): `assert_peek_victim`, `assert_recency_rank`, and `probe_resident`.

## Debugging failures

1. Re-run the failing `smoke_*` test with `--nocapture`.
2. Shrink: `PROPTEST_CASES=1 cargo test --test policy_semantics prop_<policy> -- --nocapture`.
3. Step through `model.apply(op)` vs cache state after each op.
4. Check the adapter: `Arc<V>` vs `V`, `increment_frequency` vs `touch`, dereference `&Op` keys when matching.

## Related documentation

- [Policy semantic testing (full harness)](../../docs/testing/static-analysis.md)
- [Testing strategy](../../docs/testing/testing.md)
- [Trait hierarchy](../../docs/design/trait-hierarchy.md)
- [Policy catalog](../../docs/policies/README.md)
