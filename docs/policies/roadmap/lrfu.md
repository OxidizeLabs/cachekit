# LRFU (Least Recently/Frequently Used)

## Goal

Provide a tunable continuum between recency (LRU-like) and frequency (LFU-like)
to match different workload shapes with one policy family.

## Core Idea

LRFU assigns each entry a combined recency-frequency value with exponential
decay. A tunable parameter controls how quickly older accesses lose weight:

- lower decay -> more LFU-like behavior
- higher decay -> more LRU-like behavior

Eviction selects the entry with the lowest combined value.

## Core Data Structures (Typical)

- Hash index `K -> EntryMeta`
- Per-entry state:
  - combined CRF value (combined recency-frequency)
  - last update timestamp/tick
- Victim selector ordered by current CRF value (exact or approximate)

## Complexity & Overhead

- O(log n) with exact ordered structures
- Requires per-hit value updates (or lazy recomputation) to account for decay
- Metadata and math cost are higher than plain LRU/Clock

## Notes For CacheKit

- Useful as a benchmark control policy for mapping recency-vs-frequency sensitivity.
- Prefer lazy decay application to reduce write amplification on hot paths.
- Keep numeric updates deterministic and avoid floating-point where practical.

## References

- Lee et al. (2001): “LRFU: A Spectrum of Policies that Subsumes the Least Recently Used and Least Frequently Used Policies”.
- Wikipedia (taxonomy context): [https://en.wikipedia.org/wiki/Cache_replacement_policies](https://en.wikipedia.org/wiki/Cache_replacement_policies)
