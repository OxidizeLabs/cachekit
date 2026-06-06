# FIFO operational spec

> **Spec maturity:** reference, tla
>
> **Executable oracle:** `tests/abstract_models/exact/fifo.rs` (`FifoModel`); independent reference: `reference/fifo.rs` (`NaiveFifoModel`).

First-In-First-Out cache replacement: evict the **oldest live insertion** when space is needed. This spec is independent of `FifoCache` internals; implementations must refine it.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `store` | `Set<K>` | Keys currently resident |
| `insertion_order` | `Seq<K>` | Append-only log of first-insert order (may contain stale keys) |
| `capacity` | `usize` | Maximum resident count |

**Invariant:** `store ⊆ keys(insertion_order)` (every resident key appears in the log at least once).

## Init

- `store = ∅`
- `insertion_order = ⟨⟩`
- `capacity = C` (given)

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | `store` |
| `peek_victim` | Front-to-back scan of `insertion_order`; first key still in `store`, or none if empty |
| `hit` | `Get` / `Peek` / `GetMut`: `MustHit` if key ∈ `store`, else `MustMiss` |
| `evicted_on_insert` | Key removed by eviction triggered by this `Insert`, if any |

## Operations

### `Insert(k)`

1. If `k ∈ store`: **no structural change** (value update only; queue unchanged).
2. If `capacity = 0`: no-op.
3. If `|store| ≥ capacity`: evict `peek_victim` (pop stale entries from front until a live key is removed from `store`).
4. Add `k` to `store` and append `k` to `insertion_order`.

### `Get(k)` / `Peek(k)` / `GetMut(k)`

- Set `hit` from membership in `store`.
- **No** change to `store` or `insertion_order`.

### `Touch(k)`

- Harness adapter: no-op on cache.
- Model sets `hit = MayHitOrMiss`.

### `Remove(k)`

- Remove `k` from `store` if present.
- **Do not** remove `k` from `insertion_order` (stale entries skipped on eviction).

### `EvictOne`

- If `peek_victim` is `v`: remove `v` from `store` and record `victim = Exact(v)`.
- Pop stale keys from front of `insertion_order` until the evicted key is consumed.

## Tie-breaks

- Victim: deterministic **front-to-back** scan of `insertion_order`.
- First live key wins; no randomness.

## Harness `Op` mapping

| `Op` | Cache API | Side effects |
|------|-----------|--------------|
| `Insert(k)` | `insert(k, v)` | May evict |
| `Get(k)` | `get(k)` | None (FIFO) |
| `Peek(k)` | `peek(k)` | None |
| `GetMut(k)` | — | No-op in adapter |
| `Touch(k)` | — | No-op in adapter |
| `Remove(k)` | `remove(k)` | None on queue |
| `EvictOne` | `evict_one()` | Evict oldest live |

## Formal spec (TLA+)

Machine-readable structural spec: [`Fifo.tla`](Fifo.tla). Reader guide: [tla-guide.md](tla-guide.md) (FIFO example). Run TLC: [`scripts/run-fifo-tlc.sh`](../../../scripts/run-fifo-tlc.sh) (checks `SemanticOK` invariants; manual, not CI).

## References

- Johnson, T. & Shasha, D. *2Q: A Low Overhead High Performance Buffer Management Replacement Algorithm.* (FIFO as baseline insertion-order policy.)
- CacheKit: [`FifoCache`](../../src/policy/fifo.rs) must refine this spec.
