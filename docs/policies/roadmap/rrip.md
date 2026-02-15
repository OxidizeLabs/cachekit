# RRIP / DRRIP (Re-Reference Interval Prediction)

## Goal
Scan-resistant policy with low metadata overhead, originally designed for CPU caches.

## Core Idea
Assign each cache line a re-reference prediction value (RRPV), typically 2-3 bits:
- On insert: set RRPV to distant re-reference (e.g., max value)
- On hit: set RRPV to near-immediate re-reference (e.g., 0)
- On eviction: scan for RRPV == max; if none, increment all RRPVs and retry

RRIP variants:
- **SRRIP** (Static): fixed insertion policy (distant or near-immediate)
- **BRRIP** (Bimodal): occasionally insert with near-immediate to adapt
- **DRRIP** (Dynamic): use set dueling to adaptively choose between SRRIP policies

## Core Data Structures
- Hash index `K -> Entry`
- Per-entry: `rrpv: u8` (2-3 bits typical)
- For DRRIP: small set-dueling monitors and policy selector

## Complexity & Overhead
- O(1) get (hash lookup + RRPV update)
- O(n) worst-case eviction (increment sweep), O(1) expected with proper RRPV distribution
- Very low metadata: 2-3 bits per entry
- DRRIP adds small fixed overhead for set dueling

## When To Use
- Need scan resistance with minimal metadata
- Want hardware-friendly design (bit manipulation, no complex structures)
- Workloads with scans and thrashing (DRRIP adapts automatically)
- Prefer simplicity over absolute maximum hit rate

## Implementation Notes For CacheKit
- Start with SRRIP (simpler), add DRRIP if adaptation is valuable
- 2-bit RRPV is standard (values 0-3)
- Insertion policy: distant (3) for scan resistance, or bimodal (3 with occasional 0)
- Set dueling for DRRIP: dedicate small fraction of sets to each policy, choose winner

## References
- Jaleel et al. (2010): "High Performance Cache Replacement Using Re-Reference Interval Prediction (RRIP)", ISCA 2010.
- Intel's implementation in CPU caches
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
