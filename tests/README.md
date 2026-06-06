# Test Organization

Integration tests live in `tests/` and are run with `cargo test --tests`.
The full integration gate used in CI is:

```bash
cargo test --tests --all-features
```

## Feature Requirements

| Test binary | Required features |
|---|---|
| `ttl_integration_test` | `ttl` |
| `slab_concurrency` | `concurrency` |
| `store_concurrency` | `concurrency` |
| `policy_concurrency` | `concurrency`, `policy-fifo`, `policy-lru`, `policy-s3-fifo` |
| `builder_integration_test` | default features + per-test `policy-*` cfgs |
| `policy_invariants` | per-test `policy-*` cfgs |
| `performance_regression` | default policy features |
| `lru_integration_test` | `concurrency` for concurrent smoke only |
| `*_thread_safe_wrapper` | policy feature for wrapped type |

Without `--all-features`, gated binaries are skipped (not compiled as empty crates).

## Test Files

### Builder & invariants

- **`builder_integration_test.rs`** — `CacheBuilder` / `DynCache` dispatch, peek vs get recency, validation panics, equivalence smoke
- **`policy_invariants.rs`** — cross-policy capacity-0 semantics (18 policies; S3-FIFO and FIFO exceptions documented)

### Native concurrent integration

- **`policy_concurrency.rs`** — `ConcurrentLruCache`, `ConcurrentFifoCache`, `ConcurrentS3FifoCache`
- **`slab_concurrency.rs`** — `ConcurrentSlabStore` TOCTOU and atomicity
- **`store_concurrency.rs`** — `ConcurrentHashMapStore`, `ShardedHashMapStore`, `ConcurrentHandleStore`, `ConcurrentWeightStore`

### TTL

- **`ttl_integration_test.rs`** — deterministic TTL semantics (`MockClock`), builder path, proptest reference model

### Performance

- **`performance_regression.rs`** — O(1) complexity guards (not micro-benchmarks; use `cargo bench` for those)

### Single-threaded / wrapper patterns

- **`lru_integration_test.rs`** — `LruCore` Arc/zero-copy smoke; optional single-threaded `ConcurrentLruCache` smoke
- **`lfu_thread_safe_wrapper.rs`**, **`nru_thread_safe_wrapper.rs`**, **`lru_k_thread_safe_wrapper.rs`** — external `Arc<Mutex<Policy>>` pattern (not native concurrent types)

## Capacity-0: two layers

| Layer | Behavior |
|---|---|
| `CacheBuilder::new(0).build(...)` | Panics (builder validation) |
| `Policy::new(0)` (most policies) | Honors zero; rejects inserts |
| S3-FIFO | `new(0)` panics; `try_with_ratios(0, …)` returns `Err` |
| FIFO | `new(0)` honors zero; `try_new(0)` returns `Err` |

## Shared helpers

- **`common/mod.rs`** — `all_enabled_policies()`, `exercise_dyn_cache()` (included via `mod common;`)

## Running subsets

```bash
cargo test --tests --all-features
cargo test --test builder_integration_test --all-features
cargo test --test policy_invariants --all-features
cargo test --test store_concurrency --all-features
cargo test --test policy_concurrency --all-features
cargo test --test ttl_integration_test --features ttl,policy-all
cargo test --test performance_regression
cargo test --test lfu_thread_safe_wrapper -- --ignored   # long stress only
```

## Runtime expectations

Under `--all-features`, the integration suite completes in roughly **30–45 seconds**
(dominated by unit tests in `--tests` run; integration binaries alone are much faster).
Long wrapper stress tests are `#[ignore]` by default.

## Design rationale

**Tests** verify correctness guarantees and run on every CI commit.
**Benchmarks** (`benches/`) measure performance with Criterion and are not CI gates.

See `performance_regression.rs` module docs for the complexity verification strategy.
