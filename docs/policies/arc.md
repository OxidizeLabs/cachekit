# ARC (Adaptive Replacement Cache)

## Status

**Implemented** in `src/policy/arc.rs`

## Goal

Adapt between recency and frequency automatically, without fixed partition tuning.

## Core Idea

Maintain four lists:
- `T1`: resident, recent (recency)
- `T2`: resident, frequent (frequency-ish)
- `B1`: ghost history of items evicted from `T1`
- `B2`: ghost history of items evicted from `T2`

ARC maintains a target parameter `p` that controls the balance between `T1` and `T2`.
Hits in ghost lists adjust `p`:
- hit in `B1` ⇒ increase `p` (favor recency list `T1`)
- hit in `B2` ⇒ decrease `p` (favor frequency list `T2`)

## Core Data Structures

Production ARC uses:
- Hash index mapping `K -> NonNull<Node>`
- Intrusive doubly-linked lists for `T1`, `T2`
- Ghost lists `B1`, `B2` (keys only, backed by `GhostList`)

## Key Operations (High Level)

### Get Operation

- hit in `T1` ⇒ move to `T2` head (promote to frequent)
- hit in `T2` ⇒ move within `T2` to head (update recency)
- hit in `B1` ⇒ adjust `p` upward, perform replacement, insert into `T2`, remove from `B1`
- hit in `B2` ⇒ adjust `p` downward, perform replacement, insert into `T2`, remove from `B2`
- miss ⇒ insert into `T1` and potentially evict via replacement step

### Replacement Step

Chooses victim from `T1` vs `T2` depending on `p` and where the last hit occurred:
- if `|T1| >= max(1, p)`: evict from `T1` LRU → move key to `B1`
- else: evict from `T2` LRU → move key to `B2`

### Adaptation

The parameter `p` is adjusted based on ghost hits:
- Ghost hit in `B1`: increase `p` by `δ = max(1, |B2|/|B1|)`
- Ghost hit in `B2`: decrease `p` by `δ = max(1, |B1|/|B2|)`

This adaptation mechanism allows ARC to automatically favor recency or frequency
depending on the workload's characteristics.

## Complexity & Overhead

- O(1) operations (with intrusive lists + hashmap)
- More metadata than LRU/2Q/SLRU due to ghost lists and adaptivity
- Ghost lists can hold up to `capacity` keys each (2× memory overhead in keys)

## Example Usage

```rust
use cachekit::policy::arc::ARCCore;
use cachekit::traits::CoreCache;

// Create ARC cache with 100 entry capacity
let mut cache = ARCCore::new(100);

// Insert items (go to T1 - recent list)
cache.insert("page1", "content1");
cache.insert("page2", "content2");

// First access promotes to T2 (frequent list)
assert_eq!(cache.get(&"page1"), Some(&"content1"));

// Second access keeps in T2 (MRU position)
assert_eq!(cache.get(&"page1"), Some(&"content1"));

// Check list sizes
println!("T1 len: {}, T2 len: {}", cache.t1_len(), cache.t2_len());
println!("Adaptation parameter p: {}", cache.p_value());
```

## When To Use

- **Mixed workloads** with unknown or shifting balance between recency and frequency
- **Adaptive systems** where manual tuning is impractical
- **Database buffer pools** with varying access patterns
- **File system caches** with diverse workload characteristics

## When NOT To Use

- **Simple workloads** with pure temporal locality (LRU is faster)
- **Memory-constrained** systems (ghost lists add overhead)
- **Deterministic requirements** (adaptation can make behavior harder to predict)
- **Known workload characteristics** (tuned policies like 2Q may be sufficient)

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Get | O(1) |
| Insert | O(1) amortized |
| Remove | O(1) |
| Space | O(n) entries + O(n) ghost keys |
| Scan resistance | High (via ghost lists) |
| Adaptivity | Self-tuning |

## Implementation Notes

- Uses raw pointer intrusive linked lists for T1/T2
- Ghost lists implemented using `cachekit::ds::GhostList`
- Adaptation parameter `p` starts at `capacity / 2`
- Ghost lists are bounded to `capacity` each
- Promotion from T1 to T2 on re-access
- Update in T1 promotes to T2
- Update in T2 moves to MRU

## References

- Megiddo, Modha (2003): "ARC: A Self-Tuning, Low Overhead Replacement Cache", FAST 2003.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies#Adaptive_replacement_cache_(ARC)
