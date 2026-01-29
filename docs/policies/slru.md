# SLRU (Segmented LRU)

## Goal
Reduce scan pollution by separating "probationary" entries from "protected" entries.

## Core Idea
Maintain two LRU lists:
- **Probationary** segment: new or recently demoted entries
- **Protected** segment: entries that have proven reuse

Rules (typical):
- New inserts go to probationary MRU.
- On hit in probationary: promote to protected MRU.
- On hit in protected: move to protected MRU.
- Evict from probationary LRU first; if probationary is empty/too small, evict from protected LRU.

## Core Data Structures
In `cachekit`, `src/policy/slru.rs` uses:
- `HashMap<K, NonNull<Node<K,V>>>` with segment membership + intrusive list pointers
- Two intrusive LRU lists (probationary and protected)
- Raw pointer-based doubly-linked lists for O(1) operations

## Operations

### `get(key)`
- Lookup key in index
- If in probationary: detach from probationary, attach to protected MRU
- If in protected: detach and reattach to protected MRU (move to front)
- Return value reference

### `insert(key, value)`
- If `key` exists: update value in place, preserve segment position
- Else:
  - Evict if needed (from probationary LRU or protected LRU)
  - Create new node in probationary segment
  - Insert into index and attach to probationary MRU

### Eviction Logic
- If probationary length > probationary_cap: evict from probationary LRU
- Otherwise: evict from protected LRU
- This ensures scan traffic stays in probationary and doesn't pollute protected

## Complexity & Overhead
- Lookup/update/evict: O(1)
- Memory: per-entry list pointers + hash entry + segment marker
- Slightly more overhead than plain LRU due to two lists and segment tracking

## Tuning
Requires choosing sizes for probationary/protected partitions:
- Typical ratio: 25% probationary, 75% protected
- Higher probationary ratio: more tolerance for scans, but less protection for hot set
- Lower probationary ratio: stronger protection for hot set, but may evict useful items too quickly

## Concurrency Notes
Like plain LRU, SLRU mutates shared metadata on every hit:
- Single lock around the entire structure (simple approach)
- Sharding (key->shard) for scalability
- Approximate approaches (Clock-based) when write-amplification is too high

`cachekit`'s `SlruCore` is designed for single-threaded use or external synchronization.

## Safety / Invariants (Rust)
Uses intrusive pointers (`NonNull<Node<_>>`):
- Nodes allocated on heap (Box) for stable addresses
- Every pointer update preserves list invariants
- On eviction: detach from list, remove from map, then drop
- Drop implementation cleans up all nodes

## When to Use
- **Use SLRU when:**
  - You have scan-heavy workloads that pollute plain LRU
  - You want simple scan resistance without complex frequency tracking
  - You can tolerate the tuning of probationary/protected ratio

- **Avoid SLRU when:**
  - Pure temporal locality (plain LRU is simpler and faster)
  - Need adaptive partitioning (use ARC or 2Q with ghost lists)
  - Metadata overhead is critical (use Clock or simpler policies)

## Comparison with Other Policies

| Policy | Segments | Adaptivity | Complexity |
|--------|----------|------------|------------|
| LRU    | 1        | None       | Low        |
| SLRU   | 2        | None       | Medium     |
| 2Q     | 2 + ghost| Some       | Medium     |
| ARC    | 2 + 2 ghost | Full    | High       |

SLRU sits between plain LRU and more complex adaptive policies like 2Q or ARC.

## References
- Karedla et al., "Caching Strategies to Improve Disk System Performance", Computer, 1994
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
