# &lt;Policy&gt; operational spec

> **Spec maturity:** `stub` | `reference` | `tla` (comma-separated if multiple)
>
> **Executable oracle:** `tests/abstract_models/exact/&lt;policy&gt;.rs` until an independent `reference/` model exists.

&lt;One-sentence policy definition. This spec is independent of implementation internals; implementations must refine it.&gt;

## State

| Variable | Type | Meaning |
|----------|------|---------|
| | | |

**Invariants:** (structural properties at rest)

## Init

- `capacity = C` (given)
- …

## Observables

| Observable | Definition |
|------------|------------|
| `resident` | Keys currently in cache |
| `peek_victim` | Next eviction candidate, or none if empty |
| `recency_rank(k)` | 0 = MRU (LRU-family only); undefined if absent |
| `hit` | `MustHit` / `MustMiss` / `MayHitOrMiss` per op |
| `evicted_on_insert` | Key removed by eviction triggered by this `Insert`, if any |

## Operations

### `Insert(k)`

1. …

### `Get(k)` / `GetMut(k)`

- …

### `Peek(k)`

- …
- **No promotion** (LRU-family).

### `Touch(k)`

- …

### `Remove(k)`

- …

### `EvictOne`

- …

## Tie-breaks

- Victim: …
- Equal ranks: …

## Harness `Op` mapping

| `Op` | Cache API | Side effects |
|------|-----------|--------------|
| `Insert(k)` | `insert(k, v)` | |
| `Get(k)` | `get(k)` | |
| `Peek(k)` | `peek(k)` | |
| `GetMut(k)` | — | |
| `Touch(k)` | `touch(k)` | |
| `Remove(k)` | `remove(k)` | |
| `EvictOne` | `evict_one()` | |

Align with [trait hierarchy](../../design/trait-hierarchy.md): `Peek` must not change `recency_rank` on LRU-family policies.

## Formal spec (optional)

Machine-readable spec: `&lt;Policy&gt;.tla` (if present). See [tla-guide.md](tla-guide.md).

## References

- Literature / algorithm source (if applicable)
- CacheKit: [`&lt;Impl&gt;`](../../src/policy/&lt;module&gt;.rs) must refine this spec
- Policy matrix: [matrix.md](matrix.md)
