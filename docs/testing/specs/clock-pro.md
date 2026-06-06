# Clock-PRO operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** invariant checks on [`ClockProCache`](../../src/policy/clock_pro.rs); no full `PolicyModel` yet.

Clock-PRO: hot/cold/non-resident lists with clock-style second chances. Hot/cold/non-resident structure makes exact victim prediction impractical.

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

- Structural invariants only in v1.
- Victim legal set deferred.

## References

- CacheKit: [`ClockProCache`](../../src/policy/clock_pro.rs)
- Tests: `policy_semantics/clock_pro_tests.rs`
- Bounded module: `tests/abstract_models/bounded/clock_pro.rs`
- Policy matrix: [matrix.md](matrix.md)
