# CLOCK-Pro

**Feature:** `policy-clock-pro`

## Goal
Improve Clock's scan behavior by tracking hot/cold classification and ghost entries using Clock-style hands.

## Core Idea
Maintain three conceptual groups:
- **Hot pages**: protected from eviction, frequently accessed
- **Cold pages**: resident but candidates for eviction
- **Ghost entries**: keys only, recently evicted cold pages (enables re-access detection)

Clock hands circulate and adjust status; sequential scans churn through cold pages rather than evicting hot pages.

## Core Data Structures

Implementation uses:
- `index: FxHashMap<K, usize>` mapping keys to slot index in entries buffer
- `entries: Vec<Option<Entry<K, V>>>` circular buffer of resident pages
  - Each entry has: `key`, `value`, `status` (Hot/Cold), `referenced` bit
- `ghost: Vec<Option<GhostEntry<K>>>` circular buffer of ghost keys
- `ghost_index: FxHashMap<K, usize>` for O(1) ghost lookup
- Three clock hands:
  - `hand_cold`: sweeps for cold page eviction
  - `hand_hot`: sweeps for hot page demotion (used inline during eviction)
  - `ghost_hand`: position for next ghost insertion
- `target_hot_ratio`: adaptive ratio (starts at 0.5, increases on ghost hits)

## Operations

- `get(key)`:
  - If resident: set `referenced = true`, return value
  - If in ghost: return miss (but next insert of this key → hot)
  - Otherwise: return miss

- `insert(key, value)`:
  - If key exists: update value, set `referenced = true`
  - If key was in ghost: remove from ghost, insert as HOT
  - Otherwise: insert as COLD
  - If at capacity: run eviction

- `evict()`:
  - Sweep with `hand_cold`:
    - Cold + unreferenced → evict, add key to ghost ring
    - Cold + referenced → promote to Hot, clear referenced
    - Hot (if over `max_hot` limit) → demote to Cold or clear referenced
  - Fallback: force evict at current hand position if stuck

## Scan Resistance

Clock-PRO resists scan pollution because:
1. New inserts start as cold (sequential scans only touch cold pages)
2. Cold pages need a second access to become hot
3. Hot pages are protected from eviction
4. Ghost hits boost re-accessed keys directly to hot status

## Complexity & Overhead

| Operation  | Time   | Notes                                |
|------------|--------|--------------------------------------|
| `get`      | O(1)   | Hash lookup + bit operation          |
| `insert`   | O(1)*  | *Amortized; eviction may sweep       |
| `contains` | O(1)   | Hash lookup only                     |
| `remove`   | O(1)   | Hash lookup + clear slot             |

- Memory overhead: entries buffer + ghost buffer + two hash indexes
- Ghost capacity is configurable (default = resident capacity)

## Thread Safety

- `ClockProCache`: Not thread-safe, designed for single-threaded use
- For concurrent access, wrap in external synchronization

## References
- Jiang, Chen, Zhang (2005): "CLOCK-Pro: An Effective Improvement of the CLOCK Replacement"
- Wikipedia: https://en.wikipedia.org/wiki/CLOCK-Pro
