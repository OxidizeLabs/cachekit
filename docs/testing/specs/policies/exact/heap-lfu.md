# Heap-LFU operational spec

> **Spec maturity:** reference
>
> **Executable oracle:** `tests/abstract_models/exact/heap_lfu.rs` (`HeapLfuModel`); independent reference: `reference/heap_lfu.rs` (`NaiveHeapLfuModel`).

Heap-backed LFU: evict the key with **minimum frequency**; tie-break by **key order** (`Ord`).

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `freq` | `Map<K, ℕ>` | Live frequency per resident key |
| `heap` | min-heap of `(freq, k)` | May contain stale entries; rebuilt when oversized |
| `capacity` | `usize` | Maximum resident count |

## Init

- `freq = ∅`, `heap = ∅`, `capacity = C`

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `freq` |
| `peek_victim` | Min frequency; smallest `K` by `Ord` at ties |
| `hit` | `MustHit` / `MustMiss` |

## Operations

### `Insert(k)`

1. If `k ∈ resident`: no-op for frequency (value update only in implementation).
2. Else if full: pop valid min from heap (skip stale), evict, record `evicted_on_insert`.
3. Set `freq[k] = 1`, push to heap.

### `Get(k)` / `Peek(k)`

- Set `hit`. `Get` increments frequency on hit.

### `GetMut(k)` / `Touch(k)`

- `Touch`: increment on hit. `GetMut`: no-op in adapter.

### `Remove(k)` / `EvictOne`

- Remove from `freq`; heap lazily cleaned on eviction/rebuild.

## Tie-breaks

- Equal frequency: evict **smallest `K` by `Ord`**.

## Harness notes

- Tests assert **residency only** (heap staleness makes frequency oracle fragile).
- Op strategy: `standard_op_list` (not `mfu_safe`; rebuild handles staleness).

## References

- CacheKit: [`HeapLfuCache`](../../../../src/policy/heap_lfu.rs)
- Policy matrix: [matrix.md](../../matrix.md)
