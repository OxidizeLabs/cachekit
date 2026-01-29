# LIFO (Last In, First Out)

## Goal
Evict the most recently inserted entry.

## Core Idea
LIFO uses a stack structure:
- New insertions push to top of stack
- Eviction pops from top of stack (most recent)
- Opposite of FIFO (which evicts oldest)

Think of it like a stack of plates - you remove the one you just added.

## Core Data Structures
In `cachekit`, `src/policy/lifo.rs` uses:
- `HashMap<K, V>` for O(1) lookup
- `Vec<K>` as stack for insertion order
- Top of stack = most recent insertion

## Operations

### `get(key)`
- Lookup key in HashMap
- Return value reference
- **No reordering** (unlike LRU)

### `insert(key, value)`
- If `key` exists: update value in place, no stack change
- Else:
  - If at capacity: pop from stack top (most recent)
  - Push new key to stack top
  - Insert into HashMap

### Eviction Logic
- Pop from stack top (most recently inserted)
- Remove from HashMap
- This is opposite of FIFO which pops from bottom (oldest)

## Complexity & Overhead
- Lookup/insert/evict: O(1)
- Memory: HashMap + Vec (stack)
- Low overhead, no access tracking

## When to Use

LIFO is a **niche** policy - only use when you have specific evidence that newest items are least valuable:

### Use LIFO when:
- **Temporary scratch space**: Newest items are temporary and can be discarded
- **Undo/redo buffers**: Recent operations may be undone/discarded
- **Batch processing**: Newest batch items are least important
- **Known workload pattern**: Evidence shows recent inserts won't be reused

### Avoid LIFO when:
- **General-purpose caching**: Use LRU, SLRU, or S3-FIFO
- **Temporal locality**: Recently accessed items likely to be reused (use LRU)
- **Frequency matters**: Hot items should stay (use LFU)
- **Uncertain**: FIFO is more intuitive than LIFO for most cases

## Examples

### When LIFO Makes Sense

```text
Undo Buffer Pattern:
  - User does Action A (cached)
  - User does Action B (cached)
  - User does Action C (cached)
  - User hits Undo → Action C discarded (most recent)

  LIFO is perfect: most recent action discarded first
```

### When LIFO Is Bad

```text
Typical Access Pattern:
  - Insert page A
  - Insert page B
  - Insert page C
  - Access page C again ← but it was just evicted!

  LIFO evicted the item we just added and immediately needed
  LRU/FIFO would be much better
```

## Comparison with Other Policies

| Policy | Evicts        | Good For                    | Common?     |
|--------|---------------|-----------------------------|-------------|
| FIFO   | Oldest insert | Predictable order           | Common      |
| LIFO   | Newest insert | Temporary/undo buffers      | Very rare   |
| LRU    | Least recent  | Temporal locality           | Very common |
| Random | Random        | Baseline                    | Rare        |

LIFO is the **opposite** of FIFO and much less intuitive for most use cases.

## Implementation in `cachekit`

Stack-based with Vec:
- `Vec<K>`: Stack of keys (push/pop from end)
- `HashMap<K, V>`: Value storage
- No stale entry tracking needed (always valid)

## Use Case Examples

1. **Undo/Redo Management**
   - Recent operations at top of stack
   - Undo pops most recent
   - LIFO natural fit

2. **Temporary Scratch Space**
   - Newest allocations are temporary
   - Can be discarded if needed
   - LIFO prevents evicting stable data

3. **Batch Processing**
   - Process items in batches
   - Newest batch items are speculative
   - LIFO protects earlier batch results

## Safety / Invariants (Rust)
- Stack and HashMap always in sync
- No dangling keys in stack
- Update doesn't change stack position

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
- Stack data structure: https://en.wikipedia.org/wiki/Stack_(abstract_data_type)
