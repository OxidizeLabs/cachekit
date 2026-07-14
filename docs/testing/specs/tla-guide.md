# TLA+ contributor guide

How to add optional TLA+ specs for cache policies. TLA+ checks **structural invariants on all reachable states** for a finite instance; Rust proptest checks **step-wise observables on long random traces**. They are complementary.

## When to add TLA+

| Tier | TLA+ fit |
|------|----------|
| **Exact** | Good when state is finite and victim rules are deterministic |
| **Mirror** | Possible if the mirrored DS has bounded finite state |
| **Bounded** | Defer — adaptive victims and legal sets are harder to model finitely |
| **Composed** | Defer — TTL deadlines add continuous time |

Start with a human operational spec ([template.md](template.md)) before writing TLA+.

## File layout

Human operational specs live under `policies/<tier>/`. TLA+ artifacts live under `formal/<policy>/`:

```text
docs/testing/specs/
├── policies/exact/<policy>.md   # human source of truth
├── formal/<policy>/
│   ├── <Policy>.tla             # MODULE name must match filename (case-sensitive on Linux)
│   ├── <policy>.cfg             # TLC constants, INVARIANT, CONSTRAINT
│   └── tlc.md                   # Runbook + alignment checklist
├── tla-guide.md                 # This file
└── formal/README.md             # Formal specs index
```

Run TLC via the generic wrapper:

```bash
./scripts/run-tlc.sh fifo
# or aliases:
./scripts/run-fifo-tlc.sh
./scripts/run-tlc.sh lru
./scripts/run-lru-tlc.sh
```

TLC is **manual only** — not run in CI.

## Authoring checklist

1. **Separate semantic from exploration invariants**
   - `SemanticOK` — policy truth (matches operational spec)
   - `ExplorationOK` — finiteness bounds (`MaxQueueLen`, etc.) not in the human spec

2. **Use a sentinel for optional values** — e.g. `NoVictim` instead of `NULL`.

3. **Match harness `Op` alphabet** — document omitted ops (read-only hits, value-only updates).

4. **Set `CHECK_DEADLOCK FALSE`** when exploration constraints can stall the state graph.

5. **Document alignment** in `formal/<policy>/tlc.md` with a change log when spec or `.tla` changes.

6. **macOS filename note** — TLC requires filename to match `MODULE` name. On case-insensitive volumes `Fifo.tla` may appear as `fifo.tla`; do not create duplicates.

---

## Worked example: FIFO

FIFO is the reference TLA+ pilot. Human spec: [fifo.md](policies/exact/fifo.md). Runbook: [formal/fifo/tlc.md](formal/fifo/tlc.md).

### Purpose

| Artifact | What it proves |
|----------|----------------|
| [fifo.md](policies/exact/fifo.md) | Human-readable source of truth |
| [Fifo.tla](formal/fifo/Fifo.tla) + TLC | Structural FIFO invariants on **all reachable states** for a tiny finite instance |
| `NaiveFifoModel` + proptest | Step-wise observables on **long random traces** (`u8` keys, capacity `1..=16`) |

### State variables glossary

| TLA+ | [fifo.md](policies/exact/fifo.md) | `NaiveFifoModel` |
|------|-------------------|------------------|
| `cache` | `store` | `store: HashSet<K>` |
| `queue` | `insertion_order` | `insertion_order: VecDeque<K>` |
| `Capacity` | `capacity` | `capacity: usize` |
| `OldestLive` | `peek_victim` | `oldest_key()` / `peek_victim_key()` |
| `NoVictim` | none (empty cache) | `None` |

**Stale entries:** after `RemoveKey`, removed keys stay in `queue` but not in `cache`. `OldestLive` skips them when scanning front-to-back.

### State layout (example)

```text
Capacity = 2

cache = {k2, k3}          ← residents (set)
queue = ⟨k1, k2, k3⟩      ← insertion log (sequence)
         ^stale  ^live ^live

OldestLive = k2             ← first queue element still in cache
```

### Action catalog

| TLA+ action | Harness `Op` | Enabled when | `cache'` | `queue'` |
|-------------|--------------|--------------|----------|----------|
| `InsertNew(k)` | `Insert(k)` (new key) | `k ∉ cache`, queue room / eviction succeeds | add `k`; evict oldest if full | `PopThroughVictim` + append `k` if full; else append |
| `RemoveKey(k)` | `Remove(k)` | `k ∈ cache` | remove `k` | unchanged (stale remains) |
| `EvictOldest` | `EvictOne` | cache nonempty, victim exists | remove oldest live | `PopThroughVictim` |

**Intentionally omitted:** `Get`, `Peek`, `GetMut`, `Touch` (no structural change); `Insert(k)` when `k ∈ cache` (value update only).

### Invariants

**`SemanticOK`** (policy truth — checked by TLC):

| Operator | Meaning |
|----------|---------|
| `LenBound` | `\|cache\| ≤ Capacity` |
| `CacheKeysInQueue` | every resident key appears in `queue` |
| `QueueConsistency` | every queue element ∈ `Keys` |
| `PeekVictimOK` | cache empty ↔ `OldestLive = NoVictim` |
| `VictimInCache` | when victim defined, it is in `cache` |

**`ExplorationOK`** (TLC finiteness only):

| Operator | Meaning |
|----------|---------|
| `QueueLengthBound` | `Len(queue) ≤ MaxQueueLen` |

### How to read `Fifo.tla`

1. CONSTANTS / VARIABLES
2. Init
3. Helpers — `OldestLive`, `PopThroughVictim`
4. Invariants — `SemanticOK`, `ExplorationOK`
5. Actions — `InsertNew`, `RemoveKey`, `EvictOldest`
6. Next / Spec

### Related files

| File | Role |
|------|------|
| [Fifo.tla](formal/fifo/Fifo.tla) | Machine-readable spec (module `Fifo`) |
| [fifo.cfg](formal/fifo/fifo.cfg) | TLC constants and `INVARIANT SemanticOK` |
| [formal/fifo/tlc.md](formal/fifo/tlc.md) | FIFO runbook and alignment log |
| [fifo.md](policies/exact/fifo.md) | Human operational spec |
| [scripts/run-fifo-tlc.sh](../../../scripts/run-fifo-tlc.sh) | One-command TLC runner |

---

## Worked example: LRU

Second TLA+ pilot. Human spec: [lru.md](policies/exact/lru.md). Runbook: [formal/lru/tlc.md](formal/lru/tlc.md).

### Purpose

| Artifact | What it proves |
|----------|----------------|
| [lru.md](policies/exact/lru.md) | Human-readable source of truth |
| [Lru.tla](formal/lru/Lru.tla) + TLC | Structural LRU invariants on **all reachable states** for a tiny finite instance |
| `NaiveLruModel` + cross-model / proptest | Step-wise observables on **long random traces** |

### State variables glossary

| TLA+ | [lru.md](policies/exact/lru.md) | `NaiveLruModel` |
|------|------------------|-----------------|
| `order` | `order` (MRU-first deque) | `access` timestamps (equivalent oracle) |
| `LruKey` | `peek_victim` (back of deque) | `lru_victim()` / `peek_victim_key()` |
| `Capacity` | `capacity` | `capacity: usize` |
| `NoVictim` | none (empty cache) | `None` |

Unlike FIFO, the LRU deque has **no stale entries** — `RemoveKey` filters the key from `order`.

### Action catalog

| TLA+ action | Harness `Op` | Enabled when | Effect |
|-------------|--------------|--------------|--------|
| `InsertNew(k)` | `Insert(k)` (new key) | `k ∉ Cache` | prepend `k`; evict LRU (back) if full |
| `PromoteKey(k)` | `Insert(k)` resident / `Get` / `Touch` hit | `k ∈ Cache` | move `k` to MRU (front) |
| `RemoveKey(k)` | `Remove(k)` | `k ∈ Cache` | remove `k` from deque |
| `EvictLru` | `EvictOne` | deque nonempty | drop LRU (back) |

**Intentionally omitted:** `Peek` (no promotion); `GetMut` in `LruCore` adapter (no-op).

### Invariants (`SemanticOK`)

| Operator | Meaning |
|----------|---------|
| `LenBound` | `Len(order) ≤ Capacity` |
| `NoDuplicates` | deque entries are unique |
| `OrderInKeys` | every element ∈ `Keys` |
| `PeekVictimOK` | empty ↔ `LruKey = NoVictim` |
| `VictimInCache` | victim defined ⇒ resident |

No `ExplorationOK` queue cap is needed — deque length is bounded by `Capacity`.

### Related files

| File | Role |
|------|------|
| [Lru.tla](formal/lru/Lru.tla) | Machine-readable spec (module `Lru`) |
| [lru.cfg](formal/lru/lru.cfg) | TLC constants and `INVARIANT SemanticOK` |
| [formal/lru/tlc.md](formal/lru/tlc.md) | LRU runbook and alignment log |
| [lru.md](policies/exact/lru.md) | Human operational spec |
| [scripts/run-lru-tlc.sh](../../../scripts/run-lru-tlc.sh) | One-command TLC runner |
