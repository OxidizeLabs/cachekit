# Operational policy specs

Human-readable specifications for eviction policies used as the **source of truth** for test-side oracles. See the [policy matrix](matrix.md) for tier, harness mode, and model paths per policy.

## Pipeline (all tiers)

```text
spec doc (this directory)
    → reference/ PolicyModel (optional — independent formulation)
    → exact/ PolicyModel (deque / DS-shaped oracle)
    → policy_semantics dual-run vs implementation
```

| Tier | Harness mode | Oracle |
|------|--------------|--------|
| Exact / mirror | DualRun | `exact/` `PolicyModel` vs impl |
| Exact (all policies with `reference/` rows in matrix) | CrossModel | `reference/` vs `exact/` |
| Bounded | InvariantOnly | structural invariants on impl |
| Composed (TTL) | DualRun + deadlines | `LruOccupancyModel` + TTL layer |

Cross-model tests prove `reference/` agrees with `exact/` on the same traces. Impl dual-run proves `exact/` agrees with real caches.

## Required sections (every policy spec)

Use [template.md](template.md) as the skeleton:

1. **Maturity banner** — `stub`, `reference`, and/or `tla`
2. **State variables** — abstract state at rest between operations
3. **Init** — empty cache at capacity `C`
4. **Per-`Op` transitions** — match the harness [`Op<K>`](../../../tests/abstract_models/mod.rs) alphabet
5. **Tie-breaks** — deterministic victim and rank when multiple keys qualify
6. **Observables** — `resident`, `peek_victim`, `recency_rank` (if applicable), `hit` classification
7. **API mapping** — how each `Op` maps to cache traits (`peek` must not promote on LRU, etc.)

See [trait hierarchy](../../design/trait-hierarchy.md) for `peek` vs `get` vs `touch`.

## Spec-change checklist

When a spec changes, update in order:

1. Spec doc in this directory
2. `tests/abstract_models/reference/<policy>.rs` (if reference model exists)
3. Cross-model test expectations (if behavior changed)
4. `tests/abstract_models/exact/<policy>.rs` if the exact model was wrong
5. TLA+ module and [tla-guide.md](tla-guide.md) / `*-tlc.md` alignment notes (if applicable)
6. Row in [matrix.md](matrix.md)

## Policy index

Full table: [matrix.md](matrix.md).

| Policy | Spec | Reference model | TLA+ |
|--------|------|-----------------|------|
| FIFO | [fifo.md](fifo.md) | `NaiveFifoModel` | [Fifo.tla](Fifo.tla) |
| LRU | [lru.md](lru.md) | `NaiveLruModel` | — |
| *(all others)* | [matrix.md](matrix.md) | stub / — | — |

## TLA+ (optional manual check)

FIFO includes a TLA+ pilot. **Read first:** [tla-guide.md](tla-guide.md) (FIFO worked example). TLC is **not** run in CI.

```bash
./scripts/run-fifo-tlc.sh
# or generic:
./scripts/run-tlc.sh Fifo fifo.cfg
```

Success: no `SemanticOK` violation on the bundled config. Runbook: [fifo-tlc.md](fifo-tlc.md).

**TLC vs Rust:** TLC proves `SemanticOK` on reachable states for a finite instance; proptest runs long sequential traces on `u8` keys. They are complementary.

## Related documentation

- [Policy matrix](matrix.md) — canonical index
- [Spec template](template.md) — new policy skeleton
- [TLA+ guide](tla-guide.md) — contributor guide
- [Abstract models README](../../../tests/abstract_models/README.md)
- [Policy semantic testing](../static-analysis.md)
- [Testing strategy](../testing.md)
