# CAR (Clock with Adaptive Replacement)

## Status

**Implemented** in `src/policy/car.rs`

## Goal

ARC-like adaptivity with Clock mechanics to reduce list manipulation overhead.
Hits only set a reference bit instead of moving entries in linked lists,
improving concurrency and cache-friendliness.

## Core Idea

Replace ARC's LRU lists with Clock structures plus ghost history:

- Two Clock rings for resident sets: **Recent** (seen once) and **Frequent** (repeated access)
- Reference bits approximate recency within each set
- ARC-like feedback from ghost hits adjusts `target_recent_size` (the adaptation parameter)
- On hit: set ref bit only (no list move)
- On eviction: sweep with clock hand, ref=0 evict, ref=1 clear and continue

## Core Data Structures

- `HashMap<K, usize>` for key → slot index
- Single slot array with metadata: key, value, referenced bit, `Ring` (Recent/Frequent)
- Two clock hands (`hand_recent`, `hand_frequent`) walking per-ring intrusive circular lists
- Ghost lists `ghost_recent`, `ghost_frequent` (keys only) via `GhostList<K>`

## Key Operations (High Level)

### Get Operation

- Hit in Recent or Frequent ring: set `referenced = true`, return value (no list move)
- Miss: not in cache (see insert for ghost hit handling)

### Insert Operation

- Key in cache: update value, set ref, return old value
- Ghost hit in `ghost_recent`: adapt (increase `target_recent_size`), evict if needed, insert into Frequent ring
- Ghost hit in `ghost_frequent`: adapt (decrease `target_recent_size`), evict if needed, insert into Frequent ring
- Miss: evict if full (replace step), insert into Recent ring

### Replacement Step

Loop until one entry is evicted:

- If `|Recent| > target_recent_size`: sweep the Recent ring
  - Ref=0: evict to `ghost_recent`, done
  - Ref=1: demote to Frequent ring (clear ref), continue
- If `|Recent| ≤ target_recent_size`: sweep the Frequent ring
  - Ref=0: evict to `ghost_frequent`, done
  - Ref=1: clear ref, advance hand, continue

### Adaptation

Same as ARC: ghost hit in `ghost_recent` increases `target_recent_size`;
ghost hit in `ghost_frequent` decreases it.

## Complexity & Overhead

- O(1) get (hash lookup + bit set)
- O(1) amortized insert
- More metadata than plain Clock due to ghost lists and dual rings
- Lower overhead than ARC on hits (no list moves)

## Example Usage

```rust
use cachekit::policy::car::CARCore;
use cachekit::traits::{CoreCache, ReadOnlyCache};

let mut cache = CARCore::new(100);
cache.insert("page1", "content1");
cache.insert("page2", "content2");
assert_eq!(cache.get(&"page1"), Some(&"content1"));
println!("recent: {}, frequent: {}, target: {}", cache.recent_len(), cache.frequent_len(), cache.target_recent_size());
```

## When To Use

- Mixed workloads with unknown recency/frequency balance
- Need scan resistance (ghost lists) like ARC
- Prefer lower hit overhead than ARC (bit set vs list move)
- Concurrency-friendly hit path

## When NOT To Use

- Simple temporal locality (Clock or LRU are simpler)
- Memory-constrained (ghost lists add overhead)
- Known workload where tuned 2Q/SLRU suffices

## Performance Characteristics

| Metric   | Value            |
|----------|------------------|
| Get      | O(1)             |
| Insert   | O(1) amortized   |
| Remove   | O(1)             |
| Space    | O(n) + ghost keys|
| Scan res | High             |
| Adaptivity | Self-tuning    |

## Implementation Notes

- Uses single slot array; Recent/Frequent are per-ring intrusive circular linked lists with separate hands
- Demotion from Recent to Frequent clears ref bit (per CAR paper)
- Ghost lists bounded to `capacity` each
- `target_recent_size` starts at `capacity / 2`

## References

- Bansal & Modha, "CAR: Clock with Adaptive Replacement", FAST 2004
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
