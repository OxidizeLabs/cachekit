# Operational policy specs

Human-readable specifications for eviction policies used as the **source of truth** for test-side oracles.

## Directory layout

```text
docs/testing/specs/
├── README.md, matrix.md, template.md, tla-guide.md   # hub docs
├── _includes/          # shared fragments (Op mapping, maturity levels)
├── policies/           # human operational specs by tier
│   ├── exact/          # FIFO, LRU, LFU, …
│   ├── mirror/         # Clock, 2Q, …
│   ├── bounded/        # ARC, S3-FIFO, …
│   └── composed/       # TTL
└── formal/             # TLA+ modules + TLC runbooks
    ├── fifo/
    └── lru/
```

**Canonical index:** [matrix.md](matrix.md) (maturity, harness mode, model paths per policy).

## Pipeline (all tiers)

```text
policies/<tier>/<policy>.md
    → reference/ PolicyModel (optional — independent formulation)
    → exact/ PolicyModel (deque / DS-shaped oracle)
    → policy_semantics dual-run vs implementation
```

| Tier | Harness mode | Oracle |
|------|--------------|--------|
| Exact / mirror | DualRun | `exact/` `PolicyModel` vs impl |
| Exact (policies with `reference/` rows) | CrossModel | `reference/` vs `exact/` |
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

Shared harness `Op` table: [_includes/harness-op-mapping.md](_includes/harness-op-mapping.md). Trait semantics: [trait hierarchy](../../design/trait-hierarchy.md).

## Spec-change checklist

When a spec changes, update in order:

1. Spec doc under `policies/<tier>/`
2. `tests/abstract_models/reference/<policy>.rs` (if reference model exists)
3. Cross-model test expectations (if behavior changed)
4. `tests/abstract_models/exact/<policy>.rs` if the exact model was wrong
5. `formal/<policy>/` TLA+ module and `tlc.md` alignment notes (if applicable)
6. Row in [matrix.md](matrix.md)

## TLA+ (optional manual check)

FIFO and LRU include TLA+ pilots under [formal/](formal/README.md). **Read first:** [tla-guide.md](tla-guide.md). TLC is **not** run in CI.

```bash
./scripts/run-tlc.sh fifo   # or ./scripts/run-fifo-tlc.sh
./scripts/run-tlc.sh lru    # or ./scripts/run-lru-tlc.sh
```

Success: no `SemanticOK` violation on the bundled config. Runbooks: [formal/fifo/tlc.md](formal/fifo/tlc.md), [formal/lru/tlc.md](formal/lru/tlc.md).

**TLC vs Rust:** TLC proves `SemanticOK` on reachable states for a finite instance; proptest runs long sequential traces on `u8` keys. They are complementary.

## Related documentation

- [Testing catalog](../catalog.md) — test types, current coverage, and gaps
- [Policy matrix](matrix.md) — canonical index
- [Policy specs by tier](policies/README.md)
- [Spec template](template.md) — new policy skeleton
- [TLA+ guide](tla-guide.md) — contributor guide
- [Abstract models README](../../../tests/abstract_models/README.md)
- [Policy semantic testing](../static-analysis.md)
- [Testing strategy](../testing.md)
