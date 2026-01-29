# Random Eviction

## Goal
Evict a uniformly random resident entry when capacity is reached.

## Core Idea
Select victims randomly with equal probability:
- No access pattern tracking
- No metadata per entry
- Zero maintenance cost
- Unpredictable which items are evicted

Random eviction provides a **baseline** for comparing smarter policies.

## Core Data Structures
In `cachekit`, `src/policy/random.rs` uses:
- `HashMap<K, (usize, V)>` mapping key to (index in vec, value)
- `Vec<K>` dense array of keys for O(1) random access
- Swap-remove technique for O(1) eviction

## Operations

### `get(key)`
- Lookup key in HashMap
- Return value reference
- **No reordering or metadata updates** (unlike LRU/LFU)

### `insert(key, value)`
- If `key` exists: update value in place
- Else:
  - If at capacity: evict random entry
  - Append key to Vec
  - Insert (index, value) into HashMap

### Random Eviction (O(1))
1. Generate random index: `i = rand() % len`
2. Get victim key: `victim = keys[i]`
3. Swap `keys[i]` with `keys[last]`
4. Update swapped key's index in HashMap
5. Pop last element
6. Remove victim from HashMap

Example:
```text
keys = [A, B, C, D]
Random picks index 1 (B)
Swap B with D: [A, D, C, B]
Update D's index to 1
Pop: [A, D, C]
Remove B from HashMap
```

## Complexity & Overhead
- Lookup: O(1)
- Insert/evict: O(1)
- Memory: HashMap + Vec, no per-entry metadata
- **Lowest overhead** of any cache policy

## When to Use

### Use Random when:
- **Benchmarking baseline**: Compare smarter policies against random
- **Truly random access**: No temporal or frequency locality
- **Minimal overhead critical**: Every byte/cycle counts
- **Testing infrastructure**: Verify cache behavior without policy complexity

### Avoid Random when:
- **Temporal locality exists**: Use LRU, SLRU, or S3-FIFO
- **Frequency matters**: Use LFU or Heap-LFU
- **Scan resistance needed**: Use LRU-K, 2Q, or S3-FIFO
- **Predictable performance required**: Random can evict hot items

## Performance Characteristics

Random provides **worst-case** performance bounds:
- Hit rate: Lower bound for policies with locality
- Latency: Unpredictable (can evict hot items)
- Overhead: Lowest possible (no tracking)

Any policy with access pattern awareness should beat Random on workloads with locality.

## Comparison with Other Policies

| Policy | Tracks Access | Overhead | Hit Rate (with locality) |
|--------|---------------|----------|--------------------------|
| Random | No            | Minimal  | Baseline (worst)         |
| FIFO   | Insertion     | Low      | Better                   |
| LRU    | Recency       | Medium   | Much better              |
| LFU    | Frequency     | Medium   | Much better              |

## Use Cases

1. **Benchmarking**
   - Baseline for policy comparisons
   - "Any smarter policy should beat random"

2. **Cache Infrastructure Testing**
   - Test eviction logic without policy complexity
   - Verify correctness independently of policy

3. **Random Access Workloads** (rare)
   - When access patterns are truly uniform
   - No temporal or frequency skew

4. **Minimal Overhead Required**
   - Embedded systems
   - Performance-critical paths where tracking is expensive

## Implementation Notes

The O(1) eviction uses swap-remove:
- `Vec<K>` provides O(1) random access
- HashMap stores `(index, value)` to track position
- Swapping with last element before removal keeps Vec dense
- Update index after swap maintains consistency

Alternative (simpler but O(n)):
- HashMap only
- Random eviction requires iteration (pick nth entry)

`cachekit` uses the O(1) approach for best performance.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
- Used as baseline in many cache research papers
