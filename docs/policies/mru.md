# MRU (Most Recently Used)

## Goal
Evict the most recently accessed entry.

## Implementation
If you already have an LRU structure (recency-ordered list):
- LRU victim is the tail
- MRU victim is the head

So MRU is typically “LRU, but evict from head instead of tail”.

## Notes
MRU is only beneficial for specific cyclic patterns; it is not a safe default.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
