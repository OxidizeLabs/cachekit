# MFU (Most Frequently Used)

## Goal
Evict the entry with the highest access frequency.

## Implementation
If you already maintain LFU metadata:
- LFU eviction chooses `min_freq`
- MFU eviction chooses `max_freq`

So you can implement MFU by:
- tracking `max_freq` (analogous to `min_freq`), and
- evicting from the highest-frequency bucket (with a tie-break, e.g., LRU within the bucket).

Heap-based variant:
- use a max-heap keyed by `(freq, key)` with lazy stale entries (same pattern as heap LFU).

## Notes
MFU is usually not a good general-purpose policy; it only makes sense when “most frequent so far” implies “least likely to be reused next” (some bursty workloads).

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
