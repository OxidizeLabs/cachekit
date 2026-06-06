# TTL operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** `tests/abstract_models/exact/lru.rs` (`LruOccupancyModel`) for LRU layer; TTL deadlines tested in `ttl_integration_test.rs`.

TTL (composed tier): LRU eviction semantics plus **time-to-live** expiration on entries. Not a standalone eviction policy — a decorator over an inner cache.

## Composition

```text
Expiring<InnerCache> + MockClock
    → LRU residency from LruOccupancyModel (when inner is LRU-family)
    → deadline / TtlStatus checks from integration tests
```

## State (conceptual)

| Layer | State |
|-------|-------|
| Inner (e.g. Fast-LRU) | Standard LRU state — see [lru.md](lru.md) |
| TTL | Per-key expiry instant; default TTL on insert |

## Harness contract (composed)

| Check | Where |
|-------|-------|
| LRU residency / rank | `LruOccupancyModel` (when applicable) |
| `TtlStatus::{Live, Expired, Absent}` | `ttl_integration_test.rs` |
| `insert_with_ttl`, `expire`, clock advance | `MockClock` deterministic traces |

## Per-`Op` notes

- Standard `Op` alphabet applies to the inner cache through the expiring wrapper.
- TTL-specific ops (`insert_with_ttl`, time advance) are hand-written in integration tests.

## References

- CacheKit: [`Expiring`](../../src/policy/expiring.rs), [`CacheBuilder`](../../src/builder.rs)
- Tests: [`ttl_integration_test.rs`](../../../tests/ttl_integration_test.rs)
- Base LRU spec: [lru.md](lru.md)
- Policy matrix: [matrix.md](matrix.md)
