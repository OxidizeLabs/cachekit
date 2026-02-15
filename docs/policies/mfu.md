# MFU (Most Frequently Used)

**Feature:** `policy-mfu`

## Goal
Evict the entry with the highest access frequency, opposite of LFU.

## Core Idea
MFU maintains frequency counters for all entries and evicts the one that has been accessed most frequently. This is counterintuitive for most caching scenarios but useful in specific cases where high-frequency items are less likely to be needed next.

## Data Structures

### Heap-Based MFU (`MfuCore`)

```
┌─────────────────────────────────────────────────────────────┐
│                      MfuCore<K, V>                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HashMapStore<K, V>         (value storage)          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  frequencies: HashMap<K, u64>  (freq tracking)       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  freq_heap: BinaryHeap<(u64, K)>  (max-heap)         │  │
│  │                                                       │  │
│  │    top → (15, page_1)  ← highest frequency          │  │
│  │         /           \                                │  │
│  │   (7, page_3)    (3, page_2)                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Operations

| Operation        | Complexity | Description                          |
|------------------|------------|--------------------------------------|
| `new(capacity)`  | O(1)       | Create cache                         |
| `insert(k, v)`   | O(log n)   | Insert entry, may evict max freq     |
| `get(&k)`        | O(log n)   | Get value, increment frequency       |
| `contains(&k)`   | O(1)       | Check if key exists                  |
| `len()`          | O(1)       | Number of entries                    |
| `pop_mfu()`      | O(log n)*  | Remove and return highest freq item  |
| `peek_mfu()`     | O(n)       | View highest freq item               |
| `frequency(&k)`  | O(1)       | Get frequency count for key          |

\* Amortized; may skip stale heap entries

## When to Use MFU

### Good Scenarios

1. **Burst Detection**
   - Temporary bursts of activity should be evicted
   - One-time scans that create high-frequency "noise"
   - Anti-scan protection when combined with other policies

2. **Baseline Comparisons**
   - Benchmarking against worst-case policies
   - Understanding workload characteristics
   - Academic research on cache behavior

3. **Specialized Workloads**
   - When "most frequent so far" correlates with "least needed next"
   - Bursty patterns where high frequency indicates completed activity
   - Workloads with inverse temporal locality

### Poor Scenarios

**⚠️ Warning**: MFU performs poorly for typical caching workloads!

- **General-purpose caching**: Use LRU, LFU, or S3-FIFO instead
- **Temporal locality**: MFU evicts what you just used frequently
- **Hot data retention**: Opposite behavior to what you want
- **Production systems**: Rarely makes sense as primary policy

## Implementation Details

### Max-Heap for MFU

```rust
// LFU uses min-heap (with Reverse wrapper)
freq_heap: BinaryHeap<Reverse<(u64, K)>>  // Min-heap
top → (1, A)  // Evict lowest frequency

// MFU uses max-heap (no Reverse wrapper)
freq_heap: BinaryHeap<(u64, K)>  // Max-heap
top → (99, A)  // Evict highest frequency
```

### Stale Entry Handling

Like `HeapLfuCache`, `MfuCore` uses lazy invalidation:

1. On `get`, increment frequency and push new `(freq, key)` to heap
2. Old heap entries become "stale" (outdated frequency)
3. During eviction, pop from heap and check if frequency matches
4. If stale (mismatch), discard and try next heap entry
5. Periodically rebuild heap to drop accumulated stale entries

### Heap Rebuild

```rust
const HEAP_REBUILD_FACTOR: usize = 3;

// When stale entries exceed 3x live entries, rebuild
if freq_heap.len() > store.len() * HEAP_REBUILD_FACTOR {
    rebuild_heap();  // Reconstruct from current frequencies
}
```

## Comparison with Other Policies

### MFU vs LFU

| Aspect          | LFU (Least)                  | MFU (Most)                   |
|-----------------|------------------------------|------------------------------|
| **Eviction**    | Lowest frequency             | Highest frequency            |
| **Data struct** | Min-heap                     | Max-heap                     |
| **Typical use** | General caching (keep hot)   | Burst detection (evict hot)  |
| **Performance** | Good for most workloads      | Poor for most workloads      |

### MFU vs LRU

| Aspect          | LRU                          | MFU                          |
|-----------------|------------------------------|------------------------------|
| **Tracks**      | Recency (time-based)         | Frequency (count-based)      |
| **Eviction**    | Least recently used          | Most frequently used         |
| **Complexity**  | O(1) all operations          | O(log n) insert/get          |
| **Scan resist** | No                           | Partially (evicts scans)     |

## Example Usage

```rust
use cachekit::policy::mfu::MfuCore;

// Create MFU cache
let mut cache = MfuCore::new(3);

// Insert items
cache.insert(1, 100);
cache.insert(2, 200);
cache.insert(3, 300);

// Create burst on item 1 (temporary high frequency)
for _ in 0..50 {
    cache.get(&1);
}

// Item 1 now has freq 51, others have freq 1
println!("Freq 1: {:?}", cache.frequency(&1)); // 51
println!("Freq 2: {:?}", cache.frequency(&2)); // 1
println!("Freq 3: {:?}", cache.frequency(&3)); // 1

// Insert new item - evicts item 1 (highest frequency)
cache.insert(4, 400);

assert!(!cache.contains(&1)); // Burst item evicted
assert!(cache.contains(&2));  // Low freq retained
assert!(cache.contains(&3));  // Low freq retained
assert!(cache.contains(&4));  // New item
```

## Performance Characteristics

### Time Complexity

- **Insert**: O(log n) - heap push for new frequency
- **Get**: O(log n) - heap push for updated frequency
- **Evict**: O(log n) amortized - may skip stale entries
- **Contains**: O(1) - direct HashMap lookup
- **Frequency**: O(1) - direct HashMap lookup

### Space Complexity

- O(n) for value storage (HashMapStore)
- O(n) for frequency tracking (HashMap)
- O(n) to O(3n) for heap (includes stale entries until rebuild)
- **Total**: ~3-5x overhead compared to simple HashMap

### Memory Layout

```
Per entry overhead:
- HashMap entry: ~32 bytes (key + Arc pointer)
- Frequency map: ~24 bytes (key + u64)
- Heap entries: ~24 bytes × (1-3) (with stale entries)
Total: ~80-104 bytes per cache entry
```

## Tuning Considerations

### Heap Rebuild Threshold

```rust
// Default: rebuild when stale entries exceed 3x live entries
const HEAP_REBUILD_FACTOR: usize = 3;
```

- **Lower (2)**: More frequent rebuilds, less memory overhead
- **Higher (5)**: Fewer rebuilds, more stale entries in heap
- **Trade-off**: Rebuild cost vs memory pressure

### Use Case Fit

MFU is **not** a general-purpose policy. Consider it only when:

1. You need a baseline for benchmarking
2. Your workload has proven inverse frequency-need correlation
3. You want to evict burst activity explicitly
4. You're combining with other policies in a hybrid approach

For most workloads, use **LFU**, **LRU**, **S3-FIFO**, or **SLRU** instead.

## References

- **Least Frequently Used (LFU)**: The opposite policy (evict min freq)
- **Wikipedia**: [Cache replacement policies](https://en.wikipedia.org/wiki/Cache_replacement_policies)
- **HeapLfuCache**: Similar implementation pattern, inverted heap order
