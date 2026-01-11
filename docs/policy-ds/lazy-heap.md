# Lazy Heap Index (Heap + Authoritative Map)

## What It Is
A “lazy heap” pattern keeps:
- an authoritative map of current scores, and
- a heap of `(score, key)` snapshots that may contain stale entries

On `pop()`, you discard stale heap entries until the top matches the authoritative score.

Used for:
- heap-based LFU
- OPT trace simulators (next-use position)
- TTL expiry heaps (expires_at)
- size/cost-aware eviction heaps (GDS/GDSF-like scores)

## Core Types
- `scores: HashMap<K, Score>` (authoritative)
- `heap: BinaryHeap<Reverse<(Score, K)>>` (min-heap) or max-heap depending on policy

## Operations
- `update(key, new_score)`:
  - `scores[key] = new_score`
  - `heap.push((new_score, key))`
- `pop_best()`:
  - loop pop heap:
    - if `(score, key)` matches `scores[key]`, accept it
    - else discard and continue

## Complexity
- `update`: O(log n)
- `pop_best`: O(log n) amortized, but can do multiple pops if many stale entries exist

## Bounding Stale Growth
If `heap.len()` grows much larger than `scores.len()`, rebuild:
- clear heap
- push one entry per `scores` item

This keeps memory bounded and improves worst-case pop latency.
