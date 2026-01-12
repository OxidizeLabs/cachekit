# Random

## Goal
Evict a random resident entry.

## Implementation
If you need O(1) eviction:
- Store keys in a dense `Vec<K>` plus `HashMap<K, index>` to support swap-remove:
  - pick random index `i`
  - victim = `keys[i]`
  - swap `keys[i]` with `keys[last]`, update indices, pop last
  - remove victim from value store

If you only have a `HashMap`, selecting a random entry is typically O(n) (iteration).

## Tradeoffs
- Very low metadata and maintenance cost.
- Can evict hot entries; unpredictable tail latency in hit rate.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
