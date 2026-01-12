# LIFO / FILO (Last-In, First-Out)

## Goal
Evict the most recently inserted entry first.

## Implementation
Same as FIFO but use a stack-like structure:
- `HashMap<K, V>`
- `Vec<K>` (or `VecDeque`) insertion stack

On eviction:
- pop from the back until you find a live key (if using lazy stale handling).

## Notes
LIFO is niche; only implement when you have evidence newest inserts are least likely to be reused.

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
