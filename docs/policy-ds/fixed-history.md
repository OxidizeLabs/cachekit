# Fixed-Size Access History (LRU-K Style)

## What It Is
Many policies need a bounded “last N touches” history per key (LRU-K, some admission filters, scan detection).

## Core Types
Per entry:
- `history: [u64; K]` + `len` + `cursor` (ring buffer), or
- `SmallVec<[u64; K]>`, or
- `VecDeque<u64>` with fixed capacity (simple, but more overhead)

Global:
- a monotonic tick counter (`u64`) or timestamp source

## Operations
- `record_access(now)`:
  - push `now`, dropping the oldest if full
- `access_count()`
- `k_distance()`:
  - “K-th most recent access time” (or “backward K-distance”) derived from history contents

## Implementation Notes
- For small fixed K (2..4), prefer a ring buffer in the entry metadata for fewer allocations.
- Keep the “time source” cheap and monotonic (a saturating counter is often enough for ordering).
- Define tie-break rules explicitly (e.g., how to rank entries with `< K` accesses).
