# NRU operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** `tests/abstract_models/exact/nru.rs` (`NruModel`) until an independent `reference/` model exists.

Not Recently Used: track reference bit per key in insertion order; evict first **unreferenced** key (swap-remove).

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `keys` | `Seq<K>` | Insertion order (swap-remove on eviction) |
| `referenced` | `Map<K, bool>` | Reference bit per key |
| `capacity` | `usize` | Maximum resident count |

## Init

- `keys = ⟨⟩`, `referenced = ∅`, `capacity = C`
- New inserts start **unreferenced** (`false`).

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `keys` |
| `hit` | `MustHit` / `MustMiss` |

## Operations

### `Insert(k)` (new key)

- If full: scan `keys` for first unreferenced; swap-remove and evict.
- Append `k` as unreferenced.

### `Get(k)` / `Peek(k)`

- `Get`: set `referenced[k] = true` on hit.
- `Peek`: no reference-bit change.

### `Remove(k)`

- Swap-remove `k` from `keys`.

## Harness notes

- **Tier:** mirror.
- No `EvictingCache` — op strategy `short_op_list_no_evict` (O(n) eviction scans).
- No explicit `EvictOne` in traces.

## References

- CacheKit: [`NruCache`](../../src/policy/nru.rs)
- Policy matrix: [matrix.md](matrix.md)
