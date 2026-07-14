# Fast-LRU operational spec

> **Spec maturity:** reference
>
> **Executable oracle:** `tests/abstract_models/exact/lru.rs` (`LruOccupancyModel`); shared reference: `reference/lru.rs` (`NaiveLruModel`) — same LRU semantics as [lru.md](lru.md).

Fast-LRU is LRU with the same eviction semantics as [lru.md](lru.md). This document records Fast-LRU-specific harness behavior; the deque oracle is shared with LRU.

## State

Same as LRU: MRU-first `order: Seq<K>` (or timestamp formulation in `NaiveLruModel`).

## Observables

Same as LRU: `resident`, `peek_victim` (LRU = back), `recency_rank`, `hit`.

## Operations

Same per-`Op` transitions as [lru.md](lru.md).

## Harness differences

| Aspect | Fast-LRU | LRU (`LruCore`) |
|--------|----------|-----------------|
| `GetMut(k)` | Applied via `get_mut` (promotes on hit) | No-op in adapter |
| Op strategy | `op_strategy_with_get_mut` | `standard_op_list` |
| Impl | [`FastLru`](../../../../src/policy/fast_lru.rs) | [`LruCore`](../../../../src/policy/lru.rs) |

Dual-impl equivalence: `dual_impl_tests.rs` asserts `LruCore` vs `FastLru` agree on `contains`, `peek_victim`, `recency_rank`.

## References

- Base spec: [lru.md](lru.md)
- Policy matrix: [matrix.md](../../matrix.md)
