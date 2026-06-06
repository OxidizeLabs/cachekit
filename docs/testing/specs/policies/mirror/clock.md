# Clock operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** `tests/abstract_models/exact/clock.rs` (`ClockModel`) — mirrors [`ClockRing`](../../../../src/ds/clock_ring.rs) until an independent `reference/` model exists.

Clock (second-chance): circular buffer with reference bits; evict first **unreferenced** slot on the clock hand.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `ring` | `ClockRing<K,V>` | Circular slots with reference bits |
| `hand` | index | Clock sweep position |

## Init

- Empty ring at capacity `C`.

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys present in ring slots |
| `peek_victim` | First unreferenced slot from hand (second-chance sweep) |
| `hit` | `MustHit` / `MustMiss` |

## Operations

### `Insert(k)`

- If new and full: sweep hand, clear reference bits, evict first unreferenced slot.
- Set reference bit on access/insert.

### `Get(k)` / `Peek(k)`

- `Get`: set reference bit on hit. `Peek`: no bit change.

### `EvictOne`

- Sweep and evict first unreferenced entry.

## Harness notes

- **Tier:** mirror — model wraps real `ClockRing` DS.
- Dual-impl: `ClockCache` vs `ClockRing` residency in `dual_impl_tests.rs`.

## References

- CacheKit: [`ClockCache`](../../../../src/policy/clock.rs)
- Policy matrix: [matrix.md](../../matrix.md)
