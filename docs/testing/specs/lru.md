# LRU operational spec

> **Spec maturity:** reference, tla
>
> **Executable oracle:** `tests/abstract_models/exact/lru.rs` (`LruOccupancyModel`); independent reference: `reference/lru.rs` (`NaiveLruModel`). Optional TLA+: [`Lru.tla`](Lru.tla) — run [`scripts/run-lru-tlc.sh`](../../../scripts/run-lru-tlc.sh); see [lru-tlc.md](lru-tlc.md).

Least Recently Used cache replacement: evict the key whose **most recent access is oldest** when space is needed. This spec is independent of `LruCore` / `FastLru` internals; implementations must refine it.

## State

Two equivalent formulations:

1. **Deque (MRU-first):** `order: Seq<K>` listing resident keys; front = MRU, back = LRU.
2. **Timestamps:** `access: Map<K, ℕ>` last-access time per resident key; monotonic `clock` incremented once per op that records access.

This document uses deque notation; the reference model uses timestamps.

## Init

- `order = ⟨⟩` (or `access = ∅`, `clock = 0`)
- `capacity = C`

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `order` |
| `peek_victim` | Back of `order` (LRU), or none if empty |
| `recency_rank(k)` | Index of `k` in `order` (0 = MRU); undefined if absent |
| `hit` | Per op: `MustHit` / `MustMiss` from membership |

**Timestamp recency rank:** sort resident keys by `(access[k] desc, k asc)`; rank = index of `k`.

## Operations

### `Insert(k)`

1. If `k ∈ resident`: **promote to MRU** (move to front / set `access[k] = ++clock`).
2. Else if `|resident| ≥ capacity`: evict LRU (pop back / min timestamp), record `evicted_on_insert`.
3. Insert `k` at MRU (push front / assign new timestamp).

### `Get(k)` / `GetMut(k)`

- If `k ∈ resident`: `MustHit`, promote to MRU.
- Else: `MustMiss`, no change.

### `Peek(k)`

- If `k ∈ resident`: `MustHit`.
- Else: `MustMiss`.
- **No promotion.**

### `Touch(k)`

- Same as `Get` for promotion: promote on hit, `MustMiss` if absent.

### `Remove(k)`

- Remove `k` from `order` / `access` if present. No promotion.

### `EvictOne`

- If LRU exists: remove back of `order`, `victim = Exact(lru)`.

## Tie-breaks

- **Victim:** LRU = back of deque = minimum `access[k]`.
- **Equal timestamps:** evict smallest `K` by `Ord` (monotonic clock per op normally keeps timestamps unique).
- **Recency rank ties:** break by `k asc` when sorting `(timestamp desc, key asc)`.

## Harness `Op` mapping

| `Op` | Cache API | Side effects |
|------|-----------|--------------|
| `Insert(k)` | `insert(k, v)` | Promote if resident; else insert/evict |
| `Get(k)` | `get(k)` | Promote on hit |
| `Peek(k)` | `peek(k)` | No promotion |
| `GetMut(k)` | — | No-op in adapter (LRU tests) |
| `Touch(k)` | `touch(k)` | Promote on hit |
| `Remove(k)` | `remove(k)` | Remove |
| `EvictOne` | `evict_one()` | Evict LRU |

Align with [trait hierarchy](../../design/trait-hierarchy.md): `Peek` must not change `recency_rank`.

## References

- Standard LRU semantics (Douglas & Thies, *LRU-K* and related literature).
- CacheKit: [`LruCore`](../../src/policy/lru.rs), [`FastLru`](../../src/policy/fast_lru.rs) must refine this spec.
