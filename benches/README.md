# Benchmarks

This folder contains Criterion benchmarks for LFU, LRU, and LRU-K. Results
include:
- Micro-ops: nanosecond-scale hot-path timings (single op).
- Policy-level: batched workloads with throughput (Melem/s).

## Latest run (local)

Micro-ops (ns/op):

| Cache | get_hit | insert_full | policy_only_touch |
| --- | --- | --- | --- |
| LRU | 29-30 | 140-141 | n/a |
| LRU-K | 40-42 | 194-200 | n/a |
| LFU | 86-87 | 199-201 | 66-67 |

Policy throughput (Melem/s = million operations per second):

| Cache | insert_get | eviction_churn | pop | touch_hotset |
| --- | --- | --- | --- | --- |
| LRU | ~10.6 | ~6.53 | ~17.4 | ~25.0 |
| LRU-K | ~6.9 | ~4.27 | ~11.4 | ~14.4 |
| LFU | ~6.6 | ~4.19 | ~10.2 | ~8.9 |

Workload throughput (Melem/s, 200k ops):

| Cache | uniform | hotset_90_10 | scan |
| --- | --- | --- | --- |
| LRU | ~6.9-7.0 | ~20.1-20.5 | ~7.1-7.3 |
| LRU-K | ~5.45-5.58 | ~16.4-17.4 | ~4.94-5.07 |
| LFU | ~4.66-4.69 | ~8.0-8.33 | ~4.90-4.95 |

Notes:
- Values above are from `cargo bench --bench lru`, `--bench lru_k`, and
  `--bench lfu` run in the same session.
- Throughput is derived from `Throughput::Elements(...)` in the benchmarks.
