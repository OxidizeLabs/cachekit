# SIEVE

## Goal
Simple, scan-resistant eviction with lazy promotion and competitive hit rates.

## Core Idea
Single FIFO queue with a "visited" bit per entry:
- On hit: set visited bit (no list move)
- On eviction: scan from tail (FIFO order)
  - If visited=1: clear bit, move to head, continue scan
  - If visited=0: evict immediately

Lazy promotion keeps the hot path fast (just a bit set) while eviction naturally
moves frequently-accessed items to the front during the eviction sweep.

## Core Data Structures
- Hash index `K -> Entry`
- Single circular queue/ring (FIFO-ordered)
- Per-entry `visited` bit
- Eviction hand pointer

## Complexity & Overhead
- O(1) get (hash lookup + bit set)
- O(1) amortized eviction (sweep clears bits, victim found in expected constant work)
- Lower metadata than Clock-PRO, S3-FIFO, or ARC
- No ghost history needed

## When To Use
- Need scan resistance without complex segmentation
- Want simpler alternative to S3-FIFO or Clock-PRO
- Value low metadata overhead
- Workloads with temporal locality + occasional scans

## References
- Zhang et al. (2023): "SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches", OSDI 2023.
- Paper: https://junchengyang.com/publication/nsdi24-SIEVE.pdf
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
