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

See [_includes/harness-op-mapping.md](_includes/harness-op-mapping.md) for the standard table. Adjust side effects per policy above.

## Formal spec (optional)

If present: `formal/&lt;policy&gt;/&lt;Policy&gt;.tla` + `tlc.md`. See [tla-guide.md](tla-guide.md) and [formal/README.md](formal/README.md).

## References

- Literature / algorithm source (if applicable)
- CacheKit: [`&lt;Impl&gt;`](../../../src/policy/&lt;module&gt;.rs) must refine this spec
- Policy matrix: [matrix.md](matrix.md)
- Tier index: [policies/README.md](policies/README.md)
