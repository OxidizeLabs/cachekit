# Testing Catalog

Reference for **testing types** used in CacheKit, types we do not use yet, and **coverage gaps** worth closing. For how to run tests and contributor workflows, see [Testing strategy](testing.md). For the policy semantic harness, see [Policy semantic testing](static-analysis.md) and the [policy spec matrix](specs/matrix.md).

## Overview

CacheKit combines several layers: unit tests close to implementation, property tests and semantic oracles for eviction correctness, fuzz targets for data-structure robustness, Miri for undefined-behavior smoke checks, integration tests for concurrency and composition, and Criterion benchmarks for performance analysis.

Not every testing type applies to every component. Exact eviction policies benefit most from model-based dual-run tests; adaptive policies need weaker legal-set oracles; concurrent wrappers need stress tests and thread-safety tooling; product-facing performance claims belong in benchmarks with optional regression gates.

```text
         fast / cheap                          slow / expensive
              │                                        │
  unit ───────┼── property ─── integration ─── fuzz ─── formal
              │         │              │          │
              │    policy_semantics    │     Miri / Loom
              │    (model dual-run)    │
              └──────── benchmarks / workload sim ────────┘
```

## What CacheKit uses today

| Type | Purpose | Location / entrypoint | CI |
|------|---------|----------------------|-----|
| **Unit tests** | Specific behaviors and edge cases | `#[cfg(test)] mod tests` in `src/` | Partial — filtered Miri on `--lib ds::` / `policy::`; not full `mod tests` in main test job |
| **Property tests (in-module)** | DS invariants over random inputs | `#[cfg(test)] mod property_tests` in `src/ds/`, some policies | `PROPTEST_CASES=1000 cargo test --lib property_tests` (property-tests job) |
| **Policy semantic tests** | Eviction/residency oracles vs implementations | `tests/abstract_models/`, `tests/policy_semantics/` | `cargo test --test policy_semantics --all-features`; higher case count in property-tests job |
| **Dual-impl equivalence** | Two implementations, same trace | `tests/policy_semantics/dual_impl_tests.rs` | Via `cargo test --test policy_semantics --all-features` |
| **Integration tests** | Multi-module wiring | `tests/lru_integration_test.rs`, `tests/ttl_integration_test.rs`, other `tests/*.rs` | `cargo test --tests --all-features` (main test job) |
| **Cross-policy invariants** | Library-wide consistency | `tests/policy_invariants.rs` | `cargo test --tests --all-features` |
| **Concurrency / stress** | Races and atomicity under load | `tests/*_concurrency.rs`, `tests/slab_concurrency.rs` | `cargo test --tests --all-features`; `lru_concurrency` / `slab_concurrency` need `concurrency` feature |
| **O(1) regression** | Algorithmic complexity guards | `tests/performance_regression.rs` | `cargo test --tests --all-features` |
| **Fuzz tests** | Crashes and invariant violations on mutated input | `fuzz/fuzz_targets/` (27 targets, DS layer) | PR smoke (60s/target), nightly continuous |
| **Miri** | Undefined-behavior smoke checks | Lib `ds::` / `policy::` (filtered) + `policy_semantics` `smoke_*` | Dedicated Miri job |
| **Doc tests** | Rustdoc examples compile and run | `///` examples in `src/` | `cargo test --doc --all-features` |
| **Example tests** | `examples/` compile and pass | `examples/` | `cargo test --examples --all-features` |
| **Benchmarks** | Throughput, hit rate, cross-policy comparison | `benches/`, `bench-support/` | `cargo bench` on `main` only |
| **Static analysis** | Style, lints, deps, security | clippy, rustfmt, `cargo audit`, `cargo deny` | Dedicated CI jobs |
| **Formal verification (manual)** | Reachable-state invariants on finite instances | `docs/testing/specs/formal/` (FIFO, LRU TLA+) | Not in CI; run `./scripts/run-tlc.sh` locally |
| **MSRV check** | Minimum supported Rust | `Cargo.toml` MSRV | Dedicated CI job |

### Policy semantic coverage by tier

Canonical per-policy status: [matrix.md](specs/matrix.md).

| Tier | Harness mode | Policies | Oracle strength |
|------|--------------|----------|-----------------|
| Exact | DualRun + CrossModel | FIFO, LRU, Fast-LRU, LIFO, MRU, LFU, Heap-LFU, MFU, LRU-K | Strong — independent `reference/` models for all exact-tier policies |
| Mirror | DualRun | Clock, 2Q, SLRU, NRU | Medium — `exact/` models mirror implementation shape; specs still `stub` |
| Bounded | InvariantOnly | ARC, CAR, Clock-PRO, S3-FIFO | Weak — structural invariants only; victim legal sets deferred |
| Composed | DualRun + deadlines | TTL | Medium — `LruOccupancyModel` + TTL integration tests; spec still `stub` |
| Deferred | — | Random | None — no semantic harness |

### Concurrency test coverage

Three policies ship native `Concurrent*` wrappers today (LRU, FIFO, S3-FIFO). See [Concurrency](../design/concurrency.md).

Integration concurrency tests in `tests/` cover FIFO, LFU, LRU, LRU-K, NRU, and `ConcurrentSlabStore`. Only `lru_concurrency.rs` and `slab_concurrency.rs` require the `concurrency` feature; the other `*_concurrency.rs` files run unconditionally. `lru_concurrency.rs` exercises `ConcurrentLruCache`; FIFO, LFU, and NRU use ad-hoc `Arc<Mutex<…>>` wrappers around sequential cores rather than native `Concurrent*` types. `ConcurrentS3FifoCache` has in-module tests in `src/policy/s3_fifo.rs` but no dedicated `tests/*_concurrency.rs` file.

### Performance regression coverage

`performance_regression.rs` verifies O(1) scaling for **LRU, LRU-K, LFU, Clock, and S3-FIFO** only. Other policies rely on unit tests and semantic harnesses for correctness, not complexity guards.

### Fuzz coverage

All 27 fuzz targets exercise **internal data structures** (ClockRing, SlotArena, GhostList, etc.), not end-to-end cache policy APIs. See [Fuzzing in CI/CD](fuzzing-cicd.md) and [fuzz/README.md](../../fuzz/README.md).

---

## Testing types not used (or barely used)

The tables below list other testing types common in systems and Rust libraries. **Relevance** indicates how well each type fits CacheKit.

### Correctness and semantics

| Type | What it checks | CacheKit today | Relevance |
|------|----------------|----------------|-----------|
| **Golden / snapshot tests** | Output matches a saved baseline | Not used | Low — semantic oracles are more stable for caches |
| **Model-based testing** | Random traces vs a reference model | Core of `policy_semantics` | **High** — already central |
| **Equivalence testing** | Two implementations, same behavior | `dual_impl_tests.rs` (LRU/Fast-LRU, Clock/ClockRing) | **High** — extend to more pairs |
| **Differential testing** | Output matches an external library on same traces | Cross-library benches only | **Medium** — good for hit-rate sanity, not CI today |
| **Formal verification** | All reachable states satisfy invariants | TLA+ for FIFO/LRU (manual) | **Medium** — complementary to proptest |
| **Symbolic / concolic execution** | Systematic path exploration | Not used | Low for this codebase size |
| **Mutation testing** | Tests catch injected bugs | Not used | **Medium** — validates harness strength |

### Input exploration

| Type | What it checks | CacheKit today | Relevance |
|------|----------------|----------------|-----------|
| **Property-based testing** | Invariants over random inputs | proptest throughout | **High** — already central |
| **Fuzz testing** | Panics and invariant breaks on arbitrary bytes | DS layer only | **High** — policy-level fuzz is a gap |
| **Regression corpora** | Saved minimal failing inputs | `tests/proptest-regressions/` | **Medium** — expand as failures shrink |
| **Adversarial testing** | Hash-collision DoS, untrusted keys | Documented (`KeysAreTrusted`) | **Medium** — no dedicated suite |

### Concurrency and runtime

| Type | What it checks | CacheKit today | Relevance |
|------|----------------|----------------|-----------|
| **Concurrency / stress tests** | Races under threaded load | Partial (`*_concurrency.rs`) | **High** |
| **Miri** | UB detection under interpreted execution | Curated subset | **High** — extend to TTL smoke |
| **Loom / Shuttle** | Exhaustive interleavings on small models | Not used | **High** for `Concurrent*` wrappers |
| **Sanitizer CI (TSan/ASan)** | Runtime memory and thread bugs | Not in CI | **Medium** — pairs well with concurrency tests |

### Performance and capacity

| Type | What it checks | CacheKit today | Relevance |
|------|----------------|----------------|-----------|
| **Micro-benchmarks** | Op latency and throughput | Criterion in `benches/` | **High** — already central |
| **Workload simulation** | Zipfian, scan, mixed R/W | `benches/workloads.rs` | **High** — benchmarks only |
| **Complexity / scaling tests** | O(1) vs O(n) as size grows | 5 policies in `performance_regression.rs` | **High** — expand coverage |
| **Performance regression gates** | CI fails on perf drop vs baseline | Benches on `main` only | **Medium** — optional nightly/PR gate |
| **Memory / allocation profiling** | Allocs in hot paths | Not automated | **Medium** — aligns with design goals |
| **Soak / endurance tests** | Long-run leaks and drift | Not used | **Low–medium** — nightly fuzz partially covers |

### Build, platform, and delivery

| Type | What it checks | CacheKit today | Relevance |
|------|----------------|----------------|-----------|
| **Feature-matrix testing** | All `Cargo` feature combinations | `--all-features` in CI | **High** — in place |
| **Cross-compilation** | Multiple OS/arch targets | ubuntu, macos, windows | **High** — in place |
| **Compatibility / semver tests** | Public API stable across releases | Not automated | **Medium** — as API stabilizes |
| **Serialization round-trip** | Persist and restore cache state | Design docs; limited tests | **Low** until serde ships |

### Security and robustness

| Type | What it checks | CacheKit today | Relevance |
|------|----------------|----------------|-----------|
| **Panic safety tests** | Invariants restored after panic | Partial in `ConcurrentLruCache` tests | **Medium** |
| **Boundary tests** | capacity=0, capacity=1, empty cache | Partial in `policy_invariants.rs` | **Medium** — extend cross-policy |
| **DoS-resistance tests** | Untrusted keys and hasher choice | Documented, not a dedicated suite | **Medium** |

---

## Known coverage gaps

Prioritized gaps between current practice and the testing goals in [Testing strategy](testing.md) and [Policy semantic testing](static-analysis.md).

### 1. Semantic oracle depth (highest impact)

- **Bounded policies** (ARC, CAR, Clock-PRO, S3-FIFO): `InvariantOnly` checks do not assert victim correctness. Future work: `OracleExpectation::Legal` legal-victim sets ([static-analysis.md](static-analysis.md)).
- **Mirror policies** (Clock, 2Q, SLRU, NRU): oracles mirror implementation; independent `reference/` models and `reference` spec maturity are pending.
- **Random**: no `PolicyModel` or `policy_semantics` harness ([matrix.md](specs/matrix.md)).
- **TTL**: integration tests exist; Miri smoke and composed-tier spec maturity are incomplete.
- **Path-sensitive traces**: documented as future harness work, not implemented.

### 2. Concurrency

- Only LRU, FIFO, and S3-FIFO ship `Concurrent*` wrappers; 15 policies rely on external locking ([Concurrency](../design/concurrency.md)).
- No `tests/*_concurrency.rs` integration suite for `ConcurrentS3FifoCache` (in-module tests exist in `src/policy/s3_fifo.rs`).
- No concurrency tests for Clock, ARC, CAR, 2Q, SLRU, MFU, and others.
- FIFO/LFU/NRU integration concurrency tests use `Mutex` wrappers, not native `Concurrent*` APIs.

### 3. Performance regression breadth

- O(1) guards cover 5 of ~18 policies.
- Workload hit-rate and throughput regressions are not gated on PRs (benchmarks run on `main` only).

### 4. Fuzz scope

- All targets hit DS primitives, not full `Cache` API sequences per policy.
- Likely missing DS fuzz: `expiration_index` (TTL path).

### 5. Formal verification automation

- TLA+ exists for FIFO and LRU only; TLC is manual, not CI.
- No trace-export bridge between TLC reachable states and Rust oracle traces ([formal/fifo/tlc.md](specs/formal/fifo/tlc.md)).

### 6. Cross-cutting integration

- **Weighted eviction**: unit tests in `store/weight.rs`; no integration tests ([Weighted eviction](../design/weighted-eviction.md)).
- **Builder / DynCache**: TTL builder path tested; CAR builder gap documented but not covered ([Builder and runtime dispatch](../design/builder-and-dyn-dispatch.md)).
- **Cross-policy invariants**: `policy_invariants.rs` is thin (mostly capacity-0 edge cases).
- **Dual-impl equivalence**: only two pairs today (LRU/Fast-LRU, Clock/ClockRing).

### 7. Process and documentation maturity

- Several policies have harness tests but operational specs remain `stub` (mirror, bounded, composed tiers).
- [Testing strategy](testing.md) describes in-module `property_tests` per module; most policies delegate to `policy_semantics` instead.
- Main CI **test** job runs `cargo test --tests --all-features` only; lib `mod tests` outside `property_tests` are not executed there (partial lib coverage via Miri and the property-tests job).

---

## Recommended additions (priority order)

1. **Legal-victim oracles for bounded policies** — upgrade invariant-only tests to semantic checks where victims are not unique.
2. **Independent reference models for mirror policies** — cross-model drift guards like exact-tier LRU.
3. **End-to-end policy fuzz targets** — full `Cache` op sequences, complementing DS fuzz.
4. **Expand O(1) regression** — FIFO, Fast-LRU, Heap-LFU, MFU, and other hot policies.
5. **`tests/*_concurrency.rs` for `ConcurrentS3FifoCache`** (beyond in-module tests) and Miri smoke for TTL.
6. **Random policy harness** — even a statistical residency/uniformity oracle helps benchmarking baselines.
7. **Optional nightly TLC job** — FIFO/LRU formal specs as a non-blocking check.
8. **Loom or TSan on concurrency tests** — small-model or sanitizer coverage for `Concurrent*` wrappers.

When adding a new test type, update this catalog if coverage or CI integration changes materially (including `.github/workflows/ci.yml`).

---

## Related documentation

- [Testing strategy](testing.md) — philosophy, four layers, how to run tests
- [Policy semantic testing](static-analysis.md) — harness architecture and contributor checklist
- [Policy spec matrix](specs/matrix.md) — canonical per-policy harness index
- [Fuzzing in CI/CD](fuzzing-cicd.md) — fuzz pipeline and target discovery
- [Adding fuzz targets](adding-fuzz-targets.md) — contributor guide for new fuzz targets
- [tests/README.md](../../tests/README.md) — integration test layout
- [abstract_models README](../../tests/abstract_models/README.md) — oracle models and tiers
- [Benchmarking design](../design/benchmarking.md) — benchmark layers vs regression tests
- [Concurrency](../design/concurrency.md) — `Concurrent*` coverage and gaps
