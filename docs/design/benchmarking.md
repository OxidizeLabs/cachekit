# Benchmarking

> Status: design rationale for the benchmark suite under [`benches/`](../../benches)
> and shared benchmark support under [`bench-support/`](../../bench-support).
> Companion to [`design.md`](design.md) §10 and the benchmark reference docs.

cachekit benchmarks are designed to answer cache questions, not just produce
fast-looking numbers. A cache policy can be excellent on uniform keys and weak
under scans, or fast on micro-operations and poor at preserving hit rate. The
benchmark suite therefore separates micro-operation cost, policy effectiveness,
trace-shaped workloads, reporting, and machine-readable artifacts.

## Goals

- Compare policies under workload shapes that resemble real cache traffic.
- Keep measured loops free of allocator noise and dynamic dispatch.
- Produce both human-readable reports and stable JSON artifacts.
- Preserve enough metadata to reproduce a run: git commit, branch, dirty bit,
  rustc version, host triple, CPU model, capacity, universe, operations, seed.
- Make adding a policy or workload a registry edit, not a benchmark rewrite.

## Benchmark Layers

The benchmark suite has four layers:

| Layer | Files | Purpose |
|---|---|---|
| Criterion measurements | `benches/workloads.rs`, `benches/ops.rs`, `benches/comparison.rs`, `benches/policy/*.rs` | statistically sampled latency and throughput |
| Console reports | `benches/reports.rs` | fast, readable tables without Criterion overhead |
| JSON artifact runner | `benches/runner.rs` | structured output for docs, charts, CI, historical comparison |
| Shared support crate | `bench-support/` | policy registry, workloads, metrics, JSON schema, doc renderer |

This split is deliberate. Criterion is good for micro-benchmark statistics; the
artifact runner is good for automation; console reports are good while tuning a
policy locally. No single binary is forced to serve every audience.

## Monomorphic Policy Registry

Benchmarks iterate policies through `for_each_policy!` in
[`bench-support/src/registry.rs`](../../bench-support/src/registry.rs):

```rust,ignore
for_each_policy! {
    with |policy_id, display_name, make_cache| {
        let mut cache = make_cache(CAPACITY);
        // measured workload...
    }
}
```

The macro expands to one block per concrete policy type. This avoids dynamic
dispatch in the measured loop while keeping policy iteration centralized.
`POLICIES` in the same module provides presentation metadata (stable id,
display name, chart color) for renderers and reports.

The trade-off is that adding a policy touches the macro and metadata table. A
test (`policies_metadata_matches_macro`) keeps the two from drifting. This is
the same explicit-boilerplate-over-magic choice as `DynCache`: more arms in
source, fewer surprises in hot code.

## Workload Registry

Workload definitions live in `bench-support/src/registry.rs`; generators live in
[`bench-support/src/workload.rs`](../../bench-support/src/workload.rs). The
current standard workloads cover:

- Uniform random keys for raw overhead baselines.
- Hot-set access for explicit skew.
- Sequential scan for scan-pollution stress.
- Zipfian and scrambled Zipfian for power-law access.
- Latest / recency-biased access.
- Shifting hotspots and flash crowds for adaptation.
- Composite scan-resistance mixes.

[`docs/benchmarks/workloads.md`](../benchmarks/workloads.md) is the catalog. It
also contains a large roadmap of workloads that should not be confused with
implemented cases. New workloads should land first in the support crate, then in
the docs, then in reports.

## Value Construction Discipline

`benches/runner.rs` pre-allocates one `Arc<u64>` per key in the universe and
passes a closure that returns `Arc::clone`:

```rust,ignore
fn preallocate_values() -> Vec<Arc<u64>> {
    (0..UNIVERSE).map(Arc::new).collect()
}
```

The rule is: **do not allocate values inside the measured operation loop**.
Allocating on every miss makes the benchmark measure the allocator and value
constructor, not the policy. A cheap `Arc::clone` isolates hit/miss behaviour,
eviction order, and policy metadata overhead.

This is especially important because policies store values differently:
`FastLru` stores `V` directly, while LRU / LFU / Heap-LFU use `Arc<V>` in some
paths. Pre-allocation keeps those representation differences from dominating
the benchmark.

## Artifact Schema

`bench-support/src/json_results.rs` defines the stable JSON schema for results:

- `SCHEMA_VERSION` follows semantic schema rules.
- Major bumps remove or rename required fields.
- Minor bumps add optional fields.
- Renderers accept any artifact with a matching major.

Each `BenchmarkArtifact` contains:

- `metadata`: timestamp, git commit, branch, dirty bit, rustc, host, CPU,
  benchmark config.
- `results`: rows keyed by policy, workload, and `case_id`.
- `metrics`: optional typed sections for hit rate, throughput, latency,
  eviction, scan resistance, adaptation speed.

The schema is presentation-neutral. Markdown tables and charts are rendered
later by `bench-support/src/bin/render_docs.rs`, so measurement and presentation
can evolve independently.

## Case IDs

Use `case_id::*` constants from `json_results.rs` instead of string literals:

- `hit_rate`
- `comprehensive`
- `scan_resistance`
- `adaptation`

This catches typos at compile time and prevents a result section from silently
disappearing from rendered docs. Adding a new case means adding a constant,
teaching the runner to populate it, and teaching the renderer how to display it.

## What Each Benchmark Answers

| Benchmark | Question |
|---|---|
| `ops.rs` | What is the raw cost of `get` / `insert` / policy-specific operations? |
| `workloads.rs` | Which policies preserve hit rate under standard workloads? |
| `comparison.rs` | How does cachekit compare with external crates (`lru`, `quick_cache`)? |
| `policy/*.rs` | What is the cost of each policy's unique operations? |
| `reports.rs` | What should a human inspect while tuning? |
| `runner.rs` | What should CI and docs consume? |

Do not overload one benchmark to answer all questions. If you need policy
micro-cost, use `ops.rs`; if you need hit rate under scans, use `workloads.rs`
or `runner.rs`.

## Reproducibility Rules

- Seed every workload. Default seed is 42 unless a benchmark is explicitly
  sweeping seeds.
- Record the git dirty bit. Dirty runs are useful locally but should not be
  published as release baselines without a note.
- Keep capacity, universe, and operation count visible in the artifact.
- Prefer `ScrambledZipfian` over raw `Zipfian` for cross-policy comparison when
  hardware prefetch could bias hot-key locality.
- Do not compare results across machines without CPU metadata. Tail latency and
  pointer-heavy policy cost are machine-sensitive.

## CI and Documentation Flow

The docs pipeline runs the benchmark suite, writes
`target/benchmarks/<run-id>/results.json`, and renders
`docs/benchmarks/latest/` plus charts. Release-tag snapshots live under
`docs/benchmarks/vX.Y.Z/`.

Manual workflow:

```bash
cargo bench --bench runner
./scripts/update_benchmark_docs.sh
```

The script is the high-level path for refreshing published benchmark docs. Use
individual benches (`cargo bench --bench ops`, `cargo bench --bench reports -- scan`)
while developing a policy.

## Adding a Policy to Benchmarks

1. Add the policy to `for_each_policy!` with a concrete constructor.
2. Add matching `PolicyMeta` in `POLICIES`.
3. Run the registry drift test.
4. Run `cargo bench --bench reports -- hit_rate` for a quick sanity check.
5. Run `cargo bench --bench runner` before publishing docs.

Keep constructors comparable. If one policy needs `Arc<u64>` and another stores
`u64`, choose the value shape that preserves fairness and document the exception
in the registry comment.

## Adding a Workload

1. Implement the generator in `bench-support/src/workload.rs`.
2. Add a `WorkloadCase` in the registry with stable id and display name.
3. Add docs in [`docs/benchmarks/workloads.md`](../benchmarks/workloads.md).
4. Add renderer support if the workload needs a custom section.
5. Run at least one policy family expected to behave differently (for example,
   LRU vs S3-FIFO for scan-heavy workloads).

Do not add a workload just because it is mathematically interesting. It should
answer a policy-selection question.

## Non-goals

- Benchmarks are not formal proofs of policy optimality.
- Benchmarks are not stable ABI. The JSON schema is versioned, but Criterion
  names and report formatting can change.
- Benchmarks do not hide hardware effects. They record enough metadata for the
  reader to judge them.
- Benchmarks do not replace fuzzing or invariant tests; they measure behaviour
  under selected workloads.

## See Also

- [Design overview](design.md) - §10 frames benchmarking at the principles level
- [Metrics](metrics.md) - recorder / snapshot / exporter split
- [Benchmark docs](../benchmarks/README.md)
- [Workload catalog](../benchmarks/workloads.md)
- [`bench-support/src/registry.rs`](../../bench-support/src/registry.rs)
- [`bench-support/src/json_results.rs`](../../bench-support/src/json_results.rs)
- [`benches/runner.rs`](../../benches/runner.rs)
