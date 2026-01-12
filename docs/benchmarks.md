# Benchmarks

This page links to the latest benchmark reports and the release-tag snapshots.

- [Latest benchmarks](../benchmarks/latest/)
- [Release benchmarks (by tag)](../benchmarks/)

For local runs and raw numbers, see `benches/README.md`.

## Latest run (release)

<!-- LATEST_RUN_START -->
Micro-ops (ns/op):

| Cache | get_hit | insert_full | policy_only_touch |
| --- | --- | --- | --- |
| LRU | TBD | TBD | n/a |
| LRU-K | TBD | TBD | n/a |
| LFU | TBD | TBD | TBD |

Policy throughput (Melem/s = million operations per second):

| Cache | insert_get | eviction_churn | pop | touch_hotset |
| --- | --- | --- | --- | --- |
| LRU | TBD | TBD | TBD | TBD |
| LRU-K | TBD | TBD | TBD | TBD |
| LFU | TBD | TBD | TBD | TBD |

Workload throughput (Melem/s, 200k ops):

| Cache | uniform | hotset_90_10 | scan |
| --- | --- | --- | --- |
| LRU | TBD | TBD | TBD |
| LRU-K | TBD | TBD | TBD |
| LFU | TBD | TBD | TBD |
<!-- LATEST_RUN_END -->

## Release summary

| Release | Date | Environment | Micro-ops (ns/op) | Policy throughput (Melem/s) | Workload throughput (Melem/s) | Report link |
| --- | --- | --- | --- | --- | --- | --- |
| v0.1.0 | 2025-01-12 | TBD | TBD | TBD | TBD | TBD |
