# LFU operational spec

> **Spec maturity:** reference
>
> **Executable oracle:** `tests/abstract_models/exact/lfu.rs` (`LfuModel`); independent reference: `reference/lfu.rs` (`NaiveLfuModel`).

Least Frequently Used cache replacement: evict the key with **minimum access frequency** when space is needed.

## State

| Variable | Type | Meaning |
|----------|------|---------|
| `freq` | `Map<K, ℕ>` | Access count per resident key |
| `buckets` | frequency-ordered structure | Min-frequency bucket with FIFO tie-break |
| `capacity` | `usize` | Maximum resident count |

## Init

- `freq = ∅`, `capacity = C`
- New inserts start at frequency 1.

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys in `freq` |
| `peek_victim` | Minimum frequency; FIFO order within min bucket |
| `frequency(k)` | `freq[k]` if resident |
| `hit` | `MustHit` / `MustMiss` from membership |

## Operations

### `Insert(k)`

1. If `k ∈ resident`: increment frequency (via `increment_frequency` on impl).
2. Else if full: evict min-frequency victim (FIFO tie within bucket), record `evicted_on_insert`.
3. Insert `k` at frequency 1.

### `Get(k)` / `Peek(k)`

- Set `hit`. `Get` increments frequency on hit.

### `GetMut(k)`

- No-op in harness adapter.

### `Touch(k)`

- `increment_frequency(k)` on hit (same frequency effect as `Get`).

### `Remove(k)`

- Remove `k` from frequency structure.

### `EvictOne`

- Evict min-frequency victim (FIFO tie-break).

## Tie-breaks

- **Victim:** lowest frequency; within equal frequency, **FIFO** (oldest in min bucket).
- Hand-written test `hand_written_lfu_fifo_tie_break` locks this behavior.

## Harness `Op` mapping

| `Op` | Cache API | Side effects |
|------|-----------|--------------|
| `Insert(k)` | `insert(k, v)` | Freq 1 or increment |
| `Get(k)` | `get(k)` | Increment on hit |
| `Peek(k)` | `peek(k)` | None |
| `GetMut(k)` | — | No-op |
| `Touch(k)` | `increment_frequency(k)` | Increment on hit |
| `Remove(k)` | `remove(k)` | Remove |
| `EvictOne` | `evict_one()` | Evict min freq |

**Dual-run extra check:** after each step, `cache.frequency(k) == model.frequency(k)` for all resident `k`.

## References

- CacheKit: [`LfuCache`](../../../../src/policy/lfu.rs)
- Policy matrix: [matrix.md](../../matrix.md)
