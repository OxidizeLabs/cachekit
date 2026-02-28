# SIEVE

## Goal
Provide strong scan resistance with very low metadata and update overhead.

## Core Idea
SIEVE is a simple FIFO-like policy with a one-bit filter:
- New entries are inserted as "cold".
- On access, an entry is marked "hot" via a single bit.
- Eviction scans from the FIFO head:
  - hot entry: clear bit and give it one more pass
  - cold entry: evict immediately

This creates low-cost second-chance behavior without maintaining full LRU order.

## Core Data Structures (Typical)
- Hash index `K -> EntryMeta`
- FIFO queue for insertion/eviction order
- One hot/cold bit per entry

## Complexity & Overhead
- Access path is O(1): bit set only (no relinking)
- Eviction is amortized O(1) with queue head advancement
- Metadata is minimal: queue links/indices + one bit

## Notes For CacheKit
- Aligns well with contiguous storage and index-based handles.
- Attractive for high-throughput paths where LRU relinking is too expensive.
- Benchmark against `S3-FIFO`, `Clock`, and `LRU` on scan-heavy mixes.

## References
- SIEVE policy publication and follow-up implementation notes.
- Wikipedia (taxonomy context): https://en.wikipedia.org/wiki/Cache_replacement_policies
