# 2Q operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** `tests/abstract_models/exact/two_q.rs` (`TwoQModel`) — mirrors `TwoQCore` until an independent `reference/` model exists.

2Q: two queues (probation A1 and protected Am); new keys enter probation; re-access promotes to protected; evict LRU from probation.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `probation` | `Seq<K>` | A1 queue — head = newest, tail = LRU (eviction victim) |
| `protected` | `Seq<K>` | Am queue — head = MRU, tail = LRU |
| `queues` | `Map<K, {Probation, Protected}>` | Per-key queue membership |
| `probation_cap` | `usize` | Max probation size |
| `protected_cap` | `usize` | Max protected size |

## Init

- Both queues empty; caps derived from total capacity and `a1_frac`.

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in either queue |
| `hit` | `MustHit` / `MustMiss` |

## Operations

### `Insert(k)` (new key)

- Insert at head of probation; evict probation tail if over cap; may demote protected LRU to probation.

### `Get(k)` / `Peek(k)`

- On hit in probation: promote to protected MRU.
- On hit in protected: promote to protected MRU.
- `Peek`: no promotion.

### `Remove(k)`

- Remove from whichever queue holds `k`.

## Harness notes

- **Tier:** mirror.
- No `EvictingCache` on impl — op strategy `standard_op_list_no_evict` (no `EvictOne`).
- Tests assert residency only.

## References

- Johnson & Shasha, *2Q: A Low Overhead High Performance Buffer Management Replacement Algorithm*
- CacheKit: [`TwoQCore`](../../src/policy/two_q.rs)
- Policy matrix: [matrix.md](matrix.md)
