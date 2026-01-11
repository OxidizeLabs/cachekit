# Ghost List (Key-Only Recency History)

## What It Is
A **ghost list** keeps a bounded recency history of *keys only* (no values). It’s used to adapt policies based on what was recently evicted.

Canonical use:
- ARC: `B1` and `B2` ghost lists
- CAR/CLOCK-Pro variants: “test” pages

## Core Types
- `HashSet<K>` (membership test)
- LRU-ish list of keys (often an intrusive list of nodes containing only `K`)
- `capacity: usize`

## Operations
- `contains(key) -> bool` (O(1))
- `record(key)`:
  - if already present: move-to-front
  - else insert at front, and if over capacity: evict tail key (remove from set)

## Invariants
- Set membership matches keys present in the list.
- List length ≤ capacity.

## Implementation Notes
- For performance, store ghost nodes in a slot arena and keep an index `K -> EntryId` for O(1) move-to-front.
- Ghost lists must not hold values; otherwise they become “real cache” and distort capacity accounting.
