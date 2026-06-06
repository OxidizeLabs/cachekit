# LRU TLA+ runbook

**Read first:** [tla-guide.md](tla-guide.md) (LRU worked example; glossary, invariants).

Manual optional check for [`Lru.tla`](Lru.tla) against [`lru.md`](lru.md) and `NaiveLruModel`.

## Prerequisites

- [TLA+ Toolbox](https://github.com/tlaplus/tlaplus/releases) or `tlc` on `PATH`

## Run TLC

From repo root:

```bash
./scripts/run-lru-tlc.sh
```

Or from this directory:

```bash
tlc -config lru.cfg Lru.tla
```

**Success criteria:** TLC completes with no `SemanticOK` violation on the bundled config.

### macOS filename

Module name is `Lru` — TLC requires file `Lru.tla`. On case-insensitive volumes Finder may show `lru.tla`; do not duplicate or delete.

## Re-run triggers

Re-run TLC and update the alignment log when any of these change:

- [`lru.md`](lru.md)
- [`Lru.tla`](Lru.tla)
- [`lru.cfg`](lru.cfg)
- `NaiveLruModel` promotion / eviction logic

## State-space vs trace testing

| TLC | Rust `policy_semantics` |
|-----|-------------------------|
| Explores all `Next` transitions from reachable states | Runs fixed sequential `Vec<Op<u8>>` traces |
| Finite `Keys = {k1, k2}`, `Capacity = 2` | Random keys `0..=255`, capacity `1..=16` |
| Proves `SemanticOK` on all reachable states | Proves step-wise observables via dual-run + cross-model |

TLC does **not** replace proptest. Trace-export cross-check is deferred.

## Alignment checklist

For each `Next` action, `NaiveLruModel::apply` must agree on observables:

| TLA+ action | `Op` | Resident set | `peek_victim` | Order / timestamps |
|-------------|------|--------------|---------------|-------------------|
| `InsertNew(k)` at capacity | `Insert(k)` | evict LRU, add `k` at MRU | new LRU | drop back, prepend `k` |
| `InsertNew(k)` below capacity | `Insert(k)` | add `k` | unchanged except new MRU | prepend `k` |
| `PromoteKey(k)` | `Insert(k)` resident / `Get` / `Touch` hit | unchanged | may change LRU | move `k` to front |
| `RemoveKey(k)` | `Remove(k)` | remove `k` | next LRU if `k` was LRU | filter `k` from deque |
| `EvictLru` | `EvictOne` | remove LRU | next LRU | drop back |

Semantic invariants (`PeekVictimOK`, `VictimInCache`, `NoDuplicates`) align with `peek_victim_key()` on every reachable state.

**Known limitations:**

- `Peek` not modeled (no promotion).
- `GetMut` omitted in `LruCore` adapter tests (modeled as promote in exact model only).
- Finite `Keys` in TLC vs `u8` in proptest.
- Deque formulation in TLA+; `NaiveLruModel` uses timestamps (cross-model proves equivalence).

## Alignment log

| Date | TLC result | Notes |
|------|------------|-------|
| 2026-06-06 | Pass | Initial pilot: MRU-first deque, `Keys={k1,k2}`, `Capacity=2` |
