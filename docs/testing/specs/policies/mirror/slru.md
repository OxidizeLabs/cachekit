# SLRU operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** `tests/abstract_models/exact/slru.rs` (`SlruModel`) — mirrors `SlruCore` until an independent `reference/` model exists.

Segmented LRU: probationary and protected segments; evict LRU from probationary; re-access promotes to protected MRU.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `probationary` | `Seq<K>` | Head = MRU, tail = LRU (eviction victim) |
| `protected` | `Seq<K>` | Head = MRU, tail = LRU |
| `segments` | `Map<K, {Probationary, Protected}>` | Per-key segment |
| `probationary_cap` | `usize` | Max probationary size |
| `capacity` | `usize` | Total capacity |

## Init

- Both segments empty; `probationary_cap = floor(capacity * probationary_frac)`.

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in either segment |
| `hit` | `MustHit` / `MustMiss` |

## Operations

### `Insert(k)` (new key)

- Insert at probationary MRU; evict probationary LRU if full.

### `Get(k)` / `Peek(k)`

- On hit in probationary: promote to protected MRU.
- On hit in protected: promote within protected.
- `Peek`: no promotion.

### `Remove(k)`

- Remove from segment deques and map.

## Harness notes

- **Tier:** mirror.
- No `EvictingCache` — op strategy `standard_op_list_no_evict`.
- Tests assert residency only.

## References

- CacheKit: [`SlruCore`](../../../../src/policy/slru.rs)
- Policy matrix: [matrix.md](../../matrix.md)
