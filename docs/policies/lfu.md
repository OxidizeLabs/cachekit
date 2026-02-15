# LFU (Least Frequently Used)

**Feature:** `policy-lfu`

## Goal
Evict the entry with the lowest access frequency; break ties by recency (common) or arbitrarily.

## Common Production Approach: Bucketed LFU (O(1))
The typical O(1) LFU design is:
- `HashMap<K, EntryId>` to locate entries
- Per-entry metadata: `freq`, plus pointers for an intrusive list inside its frequency bucket
- `HashMap<freq, BucketList>` (or `Vec<BucketList>` with bounded freq) storing LRU order within each frequency
- `min_freq` tracking the smallest frequency currently present

In `cachekit`, `src/policy/lfu.rs` implements LFU with O(1) eviction based on bucket + `min_freq`.

## Operations

### `get(key)`
- Return value and increment its frequency:
  1. Remove entry from old frequency bucket list.
  2. Increment `freq`.
  3. Insert into new bucket list (typically at head as MRU within that freq).
  4. If old bucket becomes empty and was `min_freq`, increment `min_freq`.

### `insert(key, value)`
- If present: update value and treat as an access (commonly increment frequency).
- Else:
  - If full: evict from `min_freq` bucket (usually the LRU within that bucket).
  - Insert new entry with `freq = 1` and set `min_freq = 1`.

### `pop_lfu()`
- Pop from the `min_freq` bucket; if it becomes empty, advance `min_freq`.

## Complexity & Overhead
- Lookup: O(1)
- Frequency bump: O(1)
- Eviction: O(1)
- Metadata: higher than LRU/FIFO due to frequency buckets and per-entry freq

## Aging / Decay (Important)
Pure LFU can keep “once-hot” items forever. To avoid this, add one of:
- Periodic global decay: divide all frequencies by 2 each interval (O(n) unless amortized)
- Time-decayed score: maintain `score = freq / f(age)` (more complex)
- Windowed LFU: only count accesses in a moving time window

See `lfu-aging.md` for common patterns.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
