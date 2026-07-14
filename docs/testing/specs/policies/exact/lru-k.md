# LRU-K operational spec

> **Spec maturity:** reference
>
> **Executable oracle:** `tests/abstract_models/exact/lru_k.rs` (`LruKModel`); independent reference: `reference/lru_k.rs` (`NaiveLruKModel`).

LRU-K: track last **K** access times per key; evict from **cold** segment using LRU; promote to **hot** after K-th access.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `cold` | `Seq<K>` | LRU-ordered cold segment (back = victim) |
| `hot` | `Seq<K>` | LRU-ordered hot segment |
| `segment` | `Map<K, {Cold, Hot}>` | Per-key segment |
| `history` | `Map<K, Seq<time>>` | Last K access timestamps (step counter) |
| `tick` | `ℕ` | Monotonic step counter |
| `k` | `usize` | History depth (parameter) |
| `capacity` | `usize` | Maximum resident count |

## Init

- Segments empty, `tick = 0`, `capacity = C`, `k` given.

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `segment` |
| `peek_victim` | Back of `cold` (LRU among cold keys) |
| `history(k)` | Last K access times for `k` |
| `hit` | `MustHit` / `MustMiss` |

## Operations

### `Insert(k)` / `Get(k)` / `Peek(k)` / `Touch(k)`

- Record access in `history`; increment `tick` on promoting ops.
- On K-th distinct access: promote `k` from cold to hot (MRU in hot).
- On insert when full: evict LRU from cold.

### `Remove(k)` / `EvictOne`

- Remove from appropriate segment and history.

## Tie-breaks

- Cold victim: LRU (back of `cold` deque).
- Hot segment: LRU ordering within hot (not eviction victim until demoted).

## Harness `Op` mapping

| `Op` | Traits asserted |
|------|-----------------|
| All | `HistoryTracking`, `EvictingCache` |

`GetMut`: no-op in adapter.

## References

- O'Neil, O'Neil & Weikum, *The LRU-K Page Replacement Algorithm*
- CacheKit: [`LruKCache`](../../../../src/policy/lru_k.rs)
- Policy matrix: [matrix.md](../../matrix.md)
