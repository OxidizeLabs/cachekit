# CAR operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** invariant checks on [`CarCore`](../../../../src/policy/car.rs); no full `PolicyModel` yet.

Clock with Adaptive Replacement (CAR): clock-based scan with adaptive hand and ghost entries. Victim is **not** uniquely determined from residency alone.

## Harness contract (InvariantOnly)

| Check | When |
|-------|------|
| `len <= capacity` | After every `Op` |
| `debug_validate_invariants()` | After every `Op` on `CarCore` |

## Per-`Op` adapter behavior

| `Op` | Effect |
|------|--------|
| `Insert(k)` | `insert(k, v)` |
| `Get(k)` | `get(k)` |
| `Peek(k)` | `peek(k)` |
| `Remove(k)` | `remove(k)` |
| `GetMut` / `Touch` / `EvictOne` | No-op in adapter |

## Observables

- Residency probed in smoke tests.
- Victim legal set deferred (`OracleExpectation::Legal`).

## References

- CacheKit: [`CarCore`](../../../../src/policy/car.rs)
- Tests: `policy_semantics/car_tests.rs`
- Bounded module: `tests/abstract_models/bounded/car.rs`
- Policy matrix: [matrix.md](../../matrix.md)
