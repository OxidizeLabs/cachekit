# S3-FIFO

**Feature:** `policy-s3-fifo`

## Goal
Achieve scan resistance with O(1) operations using simple FIFO queues and minimal metadata.

## Core Idea
S3-FIFO (Simple, Scalable, Scan-resistant FIFO) uses three FIFO queues to filter one-hit wonders and protect frequently accessed items:

- **Small (S)**: FIFO queue for newly inserted items
- **Main (M)**: FIFO queue for items that were accessed while in Small
- **Ghost (G)**: Bounded list tracking recently evicted keys (no values)

Behavior:
- New items go to Small with frequency counter = 0
- Accessing an item increments its frequency (capped at 3)
- When evicting from Small:
  - If freq > 0: promote to Main (reset freq to 0)
  - If freq == 0: evict and record key in Ghost
- When evicting from Main:
  - If freq > 0: reinsert at Main head (decrement freq)
  - If freq == 0: evict
- Ghost-guided admission: if inserting a key that's in Ghost, insert to Main (not Small)

## Core Data Structures
```text
┌─────────────────────────────────────────────────────────────────────────┐
│  index: HashMap<K, NodePtr>                                             │
│  ┌──────────┬──────────┐                                                │
│  │   Key    │ NodePtr  │                                                │
│  └──────────┴──────────┘                                                │
│                                                                         │
│  Small Queue (FIFO):   head ──► [new] ◄──► [old] ◄── tail (evict)      │
│  Main Queue (FIFO):    head ──► [hot] ◄──► [cold] ◄── tail (evict)     │
│  Ghost List:           Tracks recently evicted keys                     │
└─────────────────────────────────────────────────────────────────────────┘
```

Each node stores:
- Key, Value
- Frequency counter (2 bits, 0-3)
- Queue indicator (Small or Main)
- Doubly-linked list pointers

## Operations

### Insert
```text
insert(key, value):
  1. Key exists? → Update value, increment freq
  2. Key in Ghost? → Insert to Main (ghost-guided admission)
  3. Otherwise → Insert to Small
  4. Evict if over capacity
```

### Get
```text
get(key):
  1. Lookup in index → not found? return None
  2. Increment freq (capped at 3)
  3. Return &value
```

### Eviction
```text
evict_if_needed():
  while len > capacity:
    1. Pop from Small tail:
       - If freq > 0: promote to Main head, reset freq
       - If freq == 0: evict, record in Ghost
    2. If Small empty, pop from Main tail:
       - If freq > 0: reinsert to Main head, decrement freq
       - If freq == 0: evict
```

## Complexity & Overhead
- O(1) all operations (insert, get, evict)
- Per-entry overhead: ~2 pointers + 1 byte (freq + queue flag)
- Ghost list adds memory for tracking evicted keys (keys only, no values)

## Comparison with Other Policies

| Aspect          | S3-FIFO        | LRU            | 2Q             |
|-----------------|----------------|----------------|----------------|
| Scan resistance | Excellent      | Poor           | Good           |
| Get complexity  | O(1)           | O(1)*          | O(1)           |
| Per-access work | Increment freq | Move to head   | Maybe promote  |
| Memory overhead | Low            | Medium         | Medium         |
| Implementation  | Simple         | Complex lists  | Two lists      |

\* LRU requires pointer updates on every access

## Configuration Parameters

| Parameter     | Default | Description                                |
|---------------|---------|-------------------------------------------|
| `small_ratio` | 0.1     | Fraction of capacity for Small queue      |
| `ghost_ratio` | 0.9     | Fraction of capacity for Ghost list       |

Typical values:
- `small_ratio`: 0.1 (10%) works well for most workloads
- `ghost_ratio`: 0.5-1.0 depending on memory budget

## When to Use

**Best for:**
- CDN edge caches with mixed popularity
- Database buffer pools with occasional scans
- Web caches with long-tail access patterns
- Any workload where scans shouldn't pollute the cache

**Avoid when:**
- Access patterns are purely recency-based (use LRU)
- Strict frequency ordering is needed (use LFU)
- Memory is extremely constrained (Ghost list adds overhead)

## Example Usage

```rust
use cachekit::policy::s3_fifo::S3FifoCache;
use cachekit::traits::CoreCache;

// Create S3-FIFO cache with 100 capacity
let mut cache = S3FifoCache::new(100);

// Or with custom ratios
let mut cache = S3FifoCache::with_ratios(100, 0.1, 0.9);

// Insert items (go to Small queue)
cache.insert("page1", "content1");
cache.insert("page2", "content2");

// Access promotes frequency
cache.get(&"page1");

// Hot items survive scans
for i in 0..200 {
    cache.insert(format!("scan_{}", i), "data");
}

// "page1" likely survived (was accessed)
assert!(cache.contains(&"page1"));
```

## Using the Builder

```rust
use cachekit::builder::{CacheBuilder, CachePolicy};

let mut cache = CacheBuilder::new(100).build::<u64, String>(
    CachePolicy::S3Fifo {
        small_ratio: 0.1,
        ghost_ratio: 0.9,
    }
);
```

## References
- Yang et al., "FIFO queues are all you need for cache eviction", SOSP 2023
- https://s3fifo.com/
