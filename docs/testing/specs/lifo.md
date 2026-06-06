# LIFO operational spec

> **Spec maturity:** reference
>
> **Executable oracle:** `tests/abstract_models/exact/lifo.rs` (`LifoModel`); independent reference: `reference/lifo.rs` (`NaiveLifoModel`).

Last-In-First-Out cache replacement: evict the **most recently inserted** key (top of stack) when space is needed.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `stack` | `Seq<K>` | Back = newest (MRU of stack); all keys resident |
| `capacity` | `usize` | Maximum resident count |

## Init

- `stack = ⟨⟩`
- `capacity = C`

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `stack` |
| `peek_victim` | Back of `stack` (newest), or none if empty |
| `hit` | `MustHit` / `MustMiss` from membership |

## Operations

### `Insert(k)`

1. If `k ∈ resident`: no structural change (value update only).
2. If `capacity = 0`: no-op.
3. If `|resident| ≥ capacity`: evict back of `stack` (newest), record `evicted_on_insert`.
4. Push `k` onto back of `stack`.

### `Get(k)` / `Peek(k)` / `GetMut(k)`

- Set `hit` from membership. **No** promotion or stack reorder.

### `Touch(k)`

- Harness adapter: no-op. Model sets `hit = MayHitOrMiss`.

### `Remove(k)`

- Remove `k` from `stack` if present.

### `EvictOne`

- If stack nonempty: `victim = Exact(back)`, pop back.

## Tie-breaks

- Victim: always newest (stack back). Deterministic.

## Harness `Op` mapping

| `Op` | Cache API | Side effects |
|------|-----------|--------------|
| `Insert(k)` | `insert(k, v)` | May evict newest |
| `Get(k)` | `get(k)` | None |
| `Peek(k)` | `peek(k)` | None |
| `GetMut(k)` | — | No-op in adapter |
| `Touch(k)` | — | No-op in adapter |
| `Remove(k)` | `remove(k)` | Remove from stack |
| `EvictOne` | `evict_one()` | Evict newest |

## References

- CacheKit: [`LifoCore`](../../src/policy/lifo.rs)
- Policy matrix: [matrix.md](matrix.md)
