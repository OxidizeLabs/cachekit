# OPT / MIN (Belady’s Optimal)

## Goal
Provide the theoretical upper bound: evict the entry whose next use is farthest in the future.

## Implementation (Trace-Based)
OPT requires future knowledge, so it’s implemented as a simulator over a known access trace:
1. Preprocess trace to compute “next use” position for each access.
2. Maintain a resident set of size `C`.
3. On miss, admit the item and evict the resident item with the farthest next-use (or never used again).

Common data structures:
- Map from key -> next-use iterator/index
- Priority queue keyed by next-use position, with lazy stale entries (similar to heap LFU)

## Use In This Repo
Useful for benchmarks: comparing FIFO/LRU/LFU/LRU-K against OPT on fixed traces.

## References
- Belady (1966): “A study of replacement algorithms for a virtual-storage computer”.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
