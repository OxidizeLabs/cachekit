# CacheKit Benchmarks

This directory contains the benchmark suite for CacheKit cache policies.

## Benchmark Organization

### Criterion Benchmarks (Performance Measurement)

**`workloads.rs`** - Policy comparison across standard workloads
- Hit rate benchmarks for all policies × all workloads
- Scan resistance tests
- Adaptation speed tests
- Comprehensive benchmarks with latency/throughput
- Run: `cargo bench --bench workloads`

**`ops.rs`** - Micro-operation benchmarks
- Individual get/insert operation latency
- Measures raw policy overhead
- Run: `cargo bench --bench ops`

**`comparison.rs`** - External crate comparison
- Compares CacheKit policies against `lru` and `quick_cache` crates
- Run: `cargo bench --bench comparison`

**`policy/*.rs`** - Policy-specific benchmarks
- Unique operations for each policy
- Run: `cargo bench --bench policy_lru`, etc.

### Report Generators

**`reports.rs`** - Human-readable console reports
- Prints comparison tables for policies
- No Criterion overhead, just raw measurements
- Run: `cargo bench --bench reports -- <report_type>`
- Available reports:
  - `hit_rate` - Hit rate comparison table
  - `extended` - Extended workload comparison
  - `scan` - Scan resistance metrics
  - `adaptation` - Adaptation speed metrics
  - `detailed` - Single benchmark deep dive
  - `memory` - Memory overhead comparison
  - `comprehensive` - Full policy tables
  - `all` - Run all reports

**`runner.rs`** - JSON artifact generator
- Produces structured JSON results for automation
- Captures full environment metadata (git, rustc, CPU)
- Output: `target/benchmarks/<timestamp>/results.json`
- Run: `cargo bench --bench runner`

## Common Infrastructure

**`common/`** - Shared benchmark utilities
- `registry.rs` - Central registry of policies and workloads
- `metrics.rs` - Benchmark measurement utilities
- `workload.rs` - Workload generators
- `json_results.rs` - JSON output schema

## Quick Start

### Run hit rate comparison
```bash
cargo bench --bench reports -- hit_rate
```

### Generate JSON artifact
```bash
cargo bench --bench runner
# Output: target/benchmarks/<timestamp>/results.json
```

### Run full Criterion benchmarks
```bash
cargo bench --bench workloads
```

### View Criterion HTML reports
```bash
# After running criterion benchmarks
open target/criterion/report/index.html
```

## Benchmark Configuration

Default parameters (defined in each benchmark file):
- **Capacity**: 4,096 entries
- **Universe**: 16,384 possible keys
- **Operations**: 200,000 per benchmark
- **Seed**: 42 (for reproducibility)

## Policy Registry

All benchmarks use the central policy registry (`common/registry.rs`).

Current policies:
- **LRU** - Least Recently Used
- **LRU-K** - LRU with K-distance tracking
- **LFU** - Least Frequently Used
- **Heap-LFU** - LFU with heap-based eviction
- **Clock** - Clock/Second-Chance algorithm
- **S3-FIFO** - Simple Scalable FIFO variant
- **2Q** - Two-Queue algorithm

To add a new policy:
1. Add entry to `for_each_policy!` macro in `common/registry.rs`
2. All benchmarks automatically include it

## Workload Registry

Current standard workloads:
- `uniform` - Uniform random access
- `hotset_90_10` - 90% of accesses to 10% of keys
- `scan` - Sequential access
- `zipfian_1.0` - Zipfian distribution (α=1.0)
- `scrambled_zipf` - Zipfian with hashed keys
- `latest` - Recent keys are more popular
- `scan_resistance` - Mixed point queries and sequential scans
- `flash_crowd` - Sudden traffic spikes

Extended workloads include: `zipfian_0.8`, `shifting_hotspot`, `exponential`, `pareto`, `correlated`, `loop_small`, `working_set_churn`, `bursty`, `mixture`.

To add a new workload:
1. Add entry to `STANDARD_WORKLOADS` or `EXTENDED_WORKLOADS` in `common/registry.rs`
2. All benchmarks automatically test it

## Output Formats

### Criterion
- HTML reports in `target/criterion/`
- Statistical analysis with confidence intervals
- Automatic regression detection

### Console Reports
- Plain text tables
- Hit rate percentages
- Throughput (ops/sec)
- Latency distribution (p50, p95, p99)

### JSON Artifacts
- Structured data in `target/benchmarks/<timestamp>/results.json`
- Includes metadata (git commit, rustc version, CPU model)
- Schema version 1.0.0
- ~60KB per run, 91 results

## Best Practices

### For Performance Testing
1. Use release builds: `cargo bench`
2. Close other applications to reduce noise
3. Run multiple times and check consistency
4. Use Criterion for statistical rigor

### For Development
1. Use `reports.rs` for quick comparisons
2. Check `hit_rate` first to catch regressions
3. Use `runner.rs` for CI/CD integration

### For Release
1. Run full benchmark suite: `cargo bench`
2. Generate JSON artifact: `cargo bench --bench runner`
3. Compare with previous releases
4. Document significant changes in CHANGELOG

## Troubleshooting

### Benchmarks take too long
- Reduce `OPS` constant in benchmark file
- Run specific workloads: `cargo bench --bench workloads -- uniform`
- Use `reports.rs` instead of Criterion

### Results vary between runs
- Ensure system is idle during benchmarking
- Disable CPU frequency scaling if possible
- Run on dedicated hardware for consistent results
- Use longer warmup periods

### Adding new policies fails
- Check `for_each_policy!` macro syntax
- Verify policy implements `CoreCache<K, V>` trait
- Ensure type parameters match (some policies use `Arc<V>`, others use `V`)

## See Also

- [Benchmark Quick Start](../docs/benchmarks/QUICKSTART.md)
- [Benchmark Docs](../docs/benchmarks/README.md)
- [Workload Documentation](../docs/workloads.md)
