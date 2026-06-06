# MFU operational spec

> **Spec maturity:** reference
>
> **Executable oracle:** `tests/abstract_models/exact/mfu.rs` (`MfuModel`); independent reference: `reference/mfu.rs` (`NaiveMfuModel`).

Most Frequently Used cache replacement: evict the key with **maximum frequency** when space is needed.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `freq` | `Map<K, ℕ>` | Live frequency per resident key |
| `heap` | max-heap with sequence numbers | Stale entries possible |
| `capacity` | `usize` | Maximum resident count |

## Init

- `freq = ∅`, `capacity = C`

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `freq` |
| `peek_victim` | Max frequency; highest sequence number at ties (newest heap entry) |
| `hit` | `MustHit` / `MustMiss` |

## Operations

### `Insert(k)`

1. If `k ∈ resident`: increment frequency.
2. Else if full: evict max-frequency victim (newest heap entry at ties), record `evicted_on_insert`.
3. Insert at frequency 1.

### `Get(k)` / `Peek(k)`

- Set `hit`. `Get` increments on hit.

### `Remove(k)` / `EvictOne`

- **Skipped in proptest** — stale heap entries break `debug_validate_invariants` when keys are removed outside insert/evict path.

## Tie-breaks

- Equal max frequency: **newest heap entry** (highest sequence number) is evicted first; older entries survive.

## Harness notes

- Op strategy: `standard_op_list_mfu_safe` (no `Remove` / `EvictOne` in traces).
- Tests assert residency only.

## References

- CacheKit: [`MfuCore`](../../src/policy/mfu.rs)
- Policy matrix: [matrix.md](matrix.md)
