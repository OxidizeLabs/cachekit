# LIRS (Low Inter-reference Recency Set)

## Goal
Strong scan resistance by using **inter-reference recency** (distance between repeated touches) rather than simple last-touch recency.

## Core Idea (Very High Level)
Classify blocks as:
- **LIR**: low inter-reference recency (frequently reused) ⇒ protected
- **HIR**: high inter-reference recency (infrequently reused) ⇒ candidates for eviction

Maintain:
- `S` stack: tracks recency information to estimate inter-reference recency
- `Q` queue: resident HIR blocks; eviction occurs from `Q`

## Core Data Structures (Typical)
- Hash index `K -> EntryMeta`
- `S`: stack (often an intrusive list) with pruning rules to keep it meaningful
- `Q`: queue (often an intrusive list) for resident HIR entries

## Complexity & Overhead
- More complex than LRU/2Q; requires careful maintenance of `S` pruning invariants.

## References
- Jiang, Zhang (2002): “LIRS: An Efficient Low Inter-reference Recency Set Replacement Policy...”.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
