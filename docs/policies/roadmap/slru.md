# Segmented LRU (SLRU)

## Goal
Reduce scan pollution by separating “probation” entries from “protected” entries.

## Core Idea
Maintain two LRU lists:
- **Probationary** segment: new or recently demoted entries
- **Protected** segment: entries that have proven reuse

Rules (typical):
- New inserts go to probationary head.
- On hit in probationary: promote to protected head.
- On hit in protected: move to protected head.
- Evict from probationary tail first; if probationary is empty/too small, demote from protected tail into probationary head.

## Core Data Structures
- `HashMap<K, EntryMeta>` with segment membership + intrusive list pointers
- Two intrusive LRU lists

## Complexity & Overhead
- O(1) operations
- Requires choosing sizes for probationary/protected partitions (or a rule to bound protected)

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
