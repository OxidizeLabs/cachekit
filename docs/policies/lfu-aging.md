# Aging / Decayed LFU (Design Patterns)

LFU without aging tends to accumulate “historical winners” that stop being relevant.

Below are common, implementable aging strategies.

## 1) Periodic Global Decay (Halving)
Every `T` seconds (or every `N` accesses), apply:
- `freq[key] = max(1, freq[key] / 2)`

Tradeoffs:
- Simple conceptually.
- Naively O(n) per decay tick; must be infrequent or amortized.

Amortization approaches:
- Do decay in the background (careful with locks).
- Apply “lazy decay” by storing `(freq, epoch)` and adjusting when touched.

## 2) Epoch-Based Lazy Decay
Store per-entry:
- `freq: u32`
- `last_epoch: u32`
Global:
- `epoch: u32` increments periodically

On access:
- apply decay based on `epoch - last_epoch` (e.g., shift right by delta)
- then increment and set `last_epoch = epoch`

Tradeoffs:
- Keeps hot path O(1)
- Requires deciding how decay maps from epoch delta to a freq reduction

## 3) Windowed LFU (Count-Min Sketch + Window)
For very large keyspaces (web caches), maintain approximate counts over a window:
- probabilistic counting to reduce memory
- eviction based on estimated frequency

Tradeoffs:
- Approximate; implementation complexity rises
- Good when exact per-key counters are too expensive

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
