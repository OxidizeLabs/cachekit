# Policy operational specs

Human-readable specifications organized by harness tier. Each file is the **source of truth** for its policy's semantics.

## Layout

| Directory | Tier | Harness mode | Examples |
|-----------|------|--------------|----------|
| [exact/](exact/) | Exact | DualRun, CrossModel | FIFO, LRU, LFU |
| [mirror/](mirror/) | Mirror | DualRun | Clock, 2Q, SLRU |
| [bounded/](bounded/) | Bounded | InvariantOnly | ARC, S3-FIFO |
| [composed/](composed/) | Composed | DualRun + layer | TTL |

**Canonical index:** [matrix.md](../matrix.md) (maturity, models, op strategy, traits).

## Add a new policy

1. Copy [template.md](../template.md) into the tier directory (`policies/<tier>/<policy>.md`).
2. Implement `exact/` and optionally `reference/` models under `tests/abstract_models/`.
3. Wire `policy_semantics` tests and feature gates.
4. Append a row to [matrix.md](../matrix.md).
5. Optional TLA+: add `formal/<policy>/` per [tla-guide.md](../tla-guide.md).

Shared fragments: [_includes/](../_includes/) (harness `Op` table, maturity definitions).
