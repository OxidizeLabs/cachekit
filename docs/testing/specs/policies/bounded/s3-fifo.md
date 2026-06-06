# S3-FIFO operational spec

> **Spec maturity:** stub
>
> **Executable oracle:** invariant checks on [`S3FifoCache`](../../../../src/policy/s3_fifo.rs); no full `PolicyModel` yet.

S3-FIFO: three FIFO queues (small, main, ghost) for scan resistance. Victim selection depends on queue roles — not uniquely determined here.

## Harness contract (InvariantOnly)

| Check | When |
|-------|------|
| `len <= capacity` | After every `Op` |
| `check_invariants()` | After every `Op` (debug builds only in adapter) |

## Per-`Op` adapter behavior

| `Op` | Effect |
|------|--------|
| `Insert(k)` | `insert(k, v)` |
| `Get(k)` | `get(k)` |
| `Peek(k)` | `peek(k)` |
| `GetMut(k)` | `get_mut(k)` |
| `Remove(k)` | `remove(k)` |
| `Touch` / `EvictOne` | No-op in adapter |

## Observables

- Residency bound in smoke tests.
- Op strategy: `op_strategy_with_get_mut`.

## References

- Yang et al., *S3-FIFO: A Simple and Scalable FIFO-based Cache Admission and Eviction Policy*
- Tests: `policy_semantics/s3_fifo_tests.rs`
- Bounded module: `tests/abstract_models/bounded/s3_fifo.rs`
- Policy matrix: [matrix.md](../../matrix.md)
