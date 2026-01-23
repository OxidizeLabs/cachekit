# Benchmark Documentation

This directory contains automatically generated benchmark reports for the cachekit project.

## Structure

```
docs/benchmarks/
├── latest/               # Latest benchmark run
│   ├── index.md         # Human-readable report
│   └── results.json     # Raw data
└── v*.*.*/              # Release snapshots (future)
    ├── index.md
    └── results.json
```

## Viewing Results

### Latest Results
- **Online**: Visit [docs/benchmarks/latest/](latest/) on GitHub Pages
- **Local**: Open `latest/index.md` in your editor or browser

### Raw Data
- JSON artifact: `latest/results.json`
- Schema version: 1.0.0
- Parse with any JSON tool or the `bench-support` crate

## Generating Documentation

### Quick Update
```bash
# Run benchmarks and generate docs
./scripts/update_benchmark_docs.sh
```

### Manual Steps
```bash
# 1. Run benchmarks
cargo bench --bench runner

# 2. Render docs
cargo run --package bench-support --bin render_docs -- \
    target/benchmarks/<run-id>/results.json \
    docs/benchmarks/latest
```

### Skip Benchmarks (Re-render Only)
```bash
./scripts/update_benchmark_docs.sh --skip-bench
```

## What's Included

The generated `index.md` contains:

- **Environment Metadata**: Git commit, Rust version, CPU model
- **Configuration**: Capacity, universe size, operation count
- **Hit Rate Comparison**: All policies × all workloads
- **Throughput**: Million ops/sec for representative workloads
- **Latency P99**: Tail latency in nanoseconds
- **Scan Resistance**: Baseline, during scan, recovery scores
- **Adaptation Speed**: How quickly policies adapt to workload shifts
- **Policy Selection Guide**: Use-case recommendations

## Release Snapshots

For tagged releases (e.g., `v0.2.0`), create a snapshot:

```bash
cargo run --package bench-support --bin render_docs -- \
    target/benchmarks/<run-id>/results.json \
    docs/benchmarks/v0.2.0
```

This preserves historical performance data for comparison.

## CI/CD Integration

See `.github/workflows/` for automated benchmark publishing (future).

## Troubleshooting

### No results found
```
❌ No benchmark results found in target/benchmarks/
```
**Solution**: Run `cargo bench --bench runner` first

### Render fails
Check that the JSON artifact is valid:
```bash
python3 -m json.tool target/benchmarks/<run-id>/results.json > /dev/null
```

### Old results
The script always uses the latest `results.json` by timestamp. To use a specific run:
```bash
cargo run --package bench-support --bin render_docs -- \
    target/benchmarks/<specific-run-id>/results.json \
    docs/benchmarks/latest
```

## Schema Information

The JSON schema is defined in `bench-support/src/json_results.rs`:
- Version: 1.0.0
- Backwards compatible
- Extensible for new metrics

## Contributing

When adding new benchmark cases:
1. Update the runner (`benches/runner.rs`)
2. The renderer automatically picks up new metrics
3. Optionally update table generation in `render_docs.rs`

---

*For more details, see [Benchmark Quick Start](QUICKSTART.md)*
