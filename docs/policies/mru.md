# MRU (Most Recently Used)

## Goal
Evict the **most** recently accessed entry.

## Core Idea
MRU is the opposite of LRU:
- **LRU**: Evict the least recently used (tail of recency list)
- **MRU**: Evict the most recently used (head of recency list)

This is useful for specific cyclic or sequential access patterns where the most recently accessed item is unlikely to be accessed again soon.

## Core Data Structures
In `cachekit`, `src/policy/mru.rs` uses:
- `HashMap<K, NonNull<Node<K,V>>>` for O(1) key lookup
- Intrusive doubly-linked list (head=MRU, tail=LRU)
- Raw pointer-based nodes for O(1) operations

## Operations

### `get(key)`
- Lookup key in index
- Move node to head (MRU position) - making it **first** to evict!
- Return value reference

### `insert(key, value)`
- If `key` exists: update value in place, preserve position
- Else:
  - Evict from head (MRU) if needed
  - Create new node at head (MRU position)
  - Insert into index

### Eviction Logic
- Evict from head (MRU - most recently used!)
- This is the opposite of LRU which evicts from tail

## Complexity & Overhead
- Lookup/update/evict: O(1)
- Memory: per-entry list pointers + hash entry
- Same overhead as LRU, just different eviction point

## When to Use

MRU is **only** beneficial for specific access patterns:

### Use MRU when:
- **Cyclic patterns**: Access follows a predictable cycle (1, 2, 3, 1, 2, 3, ...)
- **Sequential scans**: Processing items once in sequence
- **Database query patterns**: Specific query plans where recently fetched pages won't be needed soon
- **You understand the workload**: MRU requires knowledge of access patterns

### Avoid MRU when:
- **General-purpose caching**: Use LRU, SLRU, or S3-FIFO instead
- **Temporal locality exists**: Recently accessed items are likely to be accessed again
- **Uncertain access patterns**: MRU can perform very poorly with unexpected access patterns
- **Default choice**: MRU is **not** a safe default - it's a specialized policy

## Examples

### When MRU Works Well

```text
Cyclic Pattern (database index scans):
  Access: Index A → Index B → Index C → Index A → Index B → Index C → ...

  MRU behavior:
    - Each access moves item to MRU (head)
    - When cache full, evict most recent (which won't be needed for 2 more accesses)
    - LRU would be worse here (would evict item about to be reused)
```

### When MRU Works Poorly

```text
Temporal Locality (typical workload):
  Access: page1, page1, page1, page2, page3, page1, page1, ...

  MRU behavior:
    - page1 accessed frequently → moves to MRU → gets evicted!
    - Terrible performance
    - LRU would be much better
```

## Implementation in `cachekit`

If you already have an LRU structure (recency-ordered list):
- LRU victim is the tail (least recently used)
- MRU victim is the head (most recently used)

So MRU is "LRU, but evict from head instead of tail".

## Comparison with Other Policies

| Policy | Eviction Point | Best For                  | Risk                        |
|--------|----------------|---------------------------|-----------------------------|
| LRU    | Tail (LRU)     | Temporal locality         | Scan pollution              |
| MRU    | Head (MRU)     | Cyclic/sequential patterns| Evicts frequently used items|
| SLRU   | Probation LRU  | Scan resistance           | Tuning required             |
| FIFO   | Oldest         | Predictable order         | No adaptation               |

## Safety / Invariants (Rust)
Uses intrusive pointers (`NonNull<Node<_>>`):
- Nodes allocated on heap (Box) for stable addresses
- Every pointer update preserves list invariants
- On eviction: detach from list, remove from map, then drop
- Drop implementation cleans up all nodes

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
