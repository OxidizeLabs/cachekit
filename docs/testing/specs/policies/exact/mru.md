# MRU operational spec

> **Spec maturity:** reference
>
> **Executable oracle:** `tests/abstract_models/exact/mru.rs` (`MruModel`); independent reference: `reference/mru.rs` (`NaiveMruModel`).

Most Recently Used cache replacement: evict the **most recently used** key (head of list) when space is needed.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `order` | `Seq<K>` | Front = MRU (eviction victim); back = LRU |
| `capacity` | `usize` | Maximum resident count |

## Init

- `order = ⟨⟩`
- `capacity = C`

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `order` |
| `peek_victim` | Not asserted in harness (no `VictimInspectable` on impl) |
| `hit` | `MustHit` / `MustMiss` from membership |

## Operations

### `Insert(k)`

1. If `k ∈ resident`: no-op for ordering (value update only in implementation).
2. Else if `|resident| ≥ capacity`: evict front (MRU), record `evicted_on_insert`.
3. Insert `k` at front.

### `Get(k)` / `GetMut(k)`

- If hit: promote to MRU. Set `hit` accordingly.

### `Peek(k)`

- Set `hit` from membership. **No promotion.**

### `Touch(k)`

- Promote on hit (same as `Get`).

### `Remove(k)`

- Remove `k` from `order` if present.

### `EvictOne`

- If MRU exists: remove front, `victim = Exact(front)`.

## Tie-breaks

- Victim on insert: head (MRU). Deterministic.

## Harness `Op` mapping

| `Op` | Cache API | Side effects |
|------|-----------|--------------|
| `Insert(k)` | `insert(k, v)` | Promote or insert/evict MRU |
| `Get(k)` | `get(k)` | Promote on hit |
| `Peek(k)` | `peek(k)` | No promotion |
| `GetMut(k)` | — | No-op in adapter |
| `Touch(k)` | `touch(k)` | Promote on hit |
| `Remove(k)` | `remove(k)` | Remove |
| `EvictOne` | `evict_one()` | Evict MRU |

## References

- CacheKit: [`MruCache`](../../../../src/policy/mru.rs)
- Policy matrix: [matrix.md](../../matrix.md)
