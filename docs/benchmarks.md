# Benchmarks

This page links to the latest benchmark reports and release-tag snapshots.

## Quick Links

- **[📊 Latest Benchmark Results](benchmarks/latest/)** - Comprehensive comparison of all policies
- **[📁 Raw JSON Data](benchmarks/latest/results.json)** - For tooling and custom analysis
- **[📖 Benchmarking Plan](benchmarking-plan.md)** - Methodology and implementation details
- **[💻 Run Locally](../benches/README.md)** - Instructions for running benchmarks

## Automated Reports

Benchmark reports are generated automatically from JSON artifacts using:
```bash
./scripts/update_benchmark_docs.sh
```

This runs the full benchmark suite and renders results to `docs/benchmarks/latest/`.

## Latest run (release)

<!-- LATEST_RUN_START -->
Micro-ops (ns/op):

| Cache | get_hit | insert_full | policy_only_touch |
| --- | --- | --- | --- |
| LRU | 30 | 147 | n/a |
| LRU-K | 47 | 191 | n/a |
| LFU | 92 | 216 | 63 |

Policy throughput (Melem/s = million operations per second):

| Cache | insert_get | eviction_churn | pop | touch_hotset |
| --- | --- | --- | --- | --- |
| LRU | 10.45 | 6.27 | 17.58 | 26.43 |
| LRU-K | 6.77 | 4.24 | 12.36 | 15.71 |
| LFU | 7.59 | 4.19 | TBD | TBD |

Workload throughput (Melem/s, 200k ops):

| Cache | uniform | hotset_90_10 | scan |
| --- | --- | --- | --- |
| LRU | 6.54 | 20.01 | 6.97 |
| LRU-K | 5.40 | 17.32 | 5.02 |
| LFU | 4.56 | 8.49 | 4.93 |
<!-- LATEST_RUN_END -->

## Release summary

| Release | Date | Environment | Micro-ops (ns/op) | Policy throughput (Melem/s) | Workload throughput (Melem/s) | Report link |
| --- | --- | --- | --- | --- | --- | --- |
| v0.1.0-alpha | 2026-01-13 | local | See “Latest run” | See “Latest run” | See “Latest run” | `target/criterion/report/index.html` |
