# ARC operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** invariant checks on [`ArcCore`](../../../../src/policy/arc.rs); no full `PolicyModel` yet.

Adaptive Replacement Cache: dynamically balances recency (T1) and frequency (T2) lists. Victim selection is **adaptive** — not uniquely determined from residency alone.

## State (implementation-shaped)

ARC maintains T1, T2, B1, B2 ghost lists and adaptation parameter `p`. Exact victim depends on list lengths and `p`.

## Harness contract (InvariantOnly)

| Check | When |
|-------|------|
| `len <= capacity` | After every `Op` |
| `debug_validate_invariants()` | After every `Op` |

## Per-`Op` adapter behavior

| `Op` | Effect |
|------|--------|
| `Insert(k)` | `insert(k, v)` |
| `Get(k)` | `get(k)` |
| `Peek(k)` | `peek(k)` |
| `Remove(k)` | `remove(k)` |
| `GetMut` / `Touch` / `EvictOne` | No-op in adapter |

## Observables

- **Residency** may be probed in smoke tests.
- **Victim:** legal set (future: `OracleExpectation::Legal`); not asserted in v1.

## Future work

- Independent reference model for adaptive victim intervals.
- Full `PolicyModel` dual-run when legal-victim oracle is defined.

## References

- Megiddo & Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache*
- Tests: `policy_semantics/arc_tests.rs`
- Bounded module: `tests/abstract_models/bounded/arc.rs`
- Policy matrix: [matrix.md](../../matrix.md)
