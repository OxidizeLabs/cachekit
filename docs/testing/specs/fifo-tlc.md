# FIFO TLA+ runbook

**Read first:** [tla-guide.md](tla-guide.md) (FIFO worked example; glossary, invariants).

Manual optional check for [`Fifo.tla`](Fifo.tla) against [`fifo.md`](fifo.md) and `NaiveFifoModel`.

## Prerequisites

- [TLA+ Toolbox](https://github.com/tlaplus/tlaplus/releases) or `tlc` on `PATH`

## Run TLC

From repo root:

```bash
./scripts/run-fifo-tlc.sh
```

Or from this directory:

```bash
tlc -config fifo.cfg Fifo.tla
```

**Success criteria:** TLC completes with no `SemanticOK` violation on the bundled config. `CHECK_DEADLOCK` is `FALSE` because `MaxQueueLen` is a TLC exploration bound (stale-queue growth can stall inserts).

### macOS filename

Module name is `Fifo` — TLC requires file `Fifo.tla`. On case-insensitive volumes Finder may show `fifo.tla`; do not duplicate or delete.

## Re-run triggers

Re-run TLC and update the alignment log when any of these change:

- [`fifo.md`](fifo.md)
- [`Fifo.tla`](Fifo.tla)
- [`fifo.cfg`](fifo.cfg)
- `OldestLive`, `PopThroughVictim`, or `NaiveFifoModel` eviction logic

## State-space vs trace testing

| TLC | Rust `policy_semantics` |
|-----|-------------------------|
| Explores all `Next` transitions from reachable states | Runs fixed sequential `Vec<Op<u8>>` traces |
| Finite `Keys = {k1, k2}`, `Capacity = 2` | Random keys `0..=255`, capacity `1..=16` |
| Proves `SemanticOK` on all reachable states (under `QueueLengthBound` constraint) | Proves step-wise observables via dual-run |

TLC does **not** replace proptest. Trace-export cross-check is deferred.

## Alignment checklist

For each `Next` action, `NaiveFifoModel::apply` must agree on observables:

| TLA+ action | `Op` | Resident set | `peek_victim` | Queue update |
|-------------|------|--------------|---------------|--------------|
| `InsertNew(k)` at capacity | `Insert(k)` | evict oldest live, then add `k` | oldest live in queue | `PopThroughVictim` then append `k` |
| `InsertNew(k)` below capacity | `Insert(k)` | add `k` | unchanged except new tail | append `k` |
| `RemoveKey(k)` | `Remove(k)` | remove `k`; queue retains stale entries | skip stale on scan | queue unchanged |
| `EvictOldest` | `EvictOne` | remove oldest live | next oldest live | `PopThroughVictim` |

Semantic invariants (`PeekVictimOK`, `VictimInCache`) align with `peek_victim_key()` on every reachable state.

**Known limitations:**

- `Get` / `Peek` not modeled (structural no-ops on FIFO).
- Finite `Keys` in TLC vs `u8` in proptest.
- Resident re-`Insert` omitted (`InsertNew` requires `k \notin cache`).

## Alignment log

| Date | TLC result | Notes |
|------|------------|-------|
| 2026-06-06 | Pass | Initial pilot: `PopThroughVictim`, `CacheKeysInQueue`, `Keys={k1,k2}` |
| 2026-06-06 | Pass | Hardening: `SemanticOK`/`ExplorationOK`, `PeekVictimOK`, `fifo-tla-guide.md`, `run-fifo-tlc.sh` |
