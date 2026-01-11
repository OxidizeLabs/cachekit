# ARC (Adaptive Replacement Cache)

## Goal
Adapt between recency and frequency automatically, without fixed partition tuning.

## Core Idea
Maintain four lists:
- `T1`: resident, recent (recency)
- `T2`: resident, frequent (frequency-ish)
- `B1`: ghost history of items evicted from `T1`
- `B2`: ghost history of items evicted from `T2`

ARC maintains a target parameter `p` that controls the balance between `T1` and `T2`.
Hits in ghost lists adjust `p`:
- hit in `B1` ⇒ increase `p` (favor recency list `T1`)
- hit in `B2` ⇒ decrease `p` (favor frequency list `T2`)

## Core Data Structures
Production ARC typically uses:
- Hash index mapping `K -> { which_list, node_ptr }`
- Intrusive LRU lists for `T1`, `T2`, plus ghost lists `B1`, `B2` (ghost lists store keys only)

## Key Operations (High Level)
- `get`:
  - hit in `T1` ⇒ move to `T2` head
  - hit in `T2` ⇒ move within `T2` to head
  - hit in `B1`/`B2` ⇒ adjust `p`, perform replacement step, move into `T2`
  - miss ⇒ insert into `T1` and potentially evict via replacement step

Replacement step chooses victim from `T1` vs `T2` depending on `p` and where the last hit occurred.

## Complexity & Overhead
- O(1) operations (with intrusive lists + hashmap)
- More metadata and code complexity than LRU/2Q/SLRU due to ghost lists and adaptivity

## References
- Megiddo, Modha (2003): “ARC: A Self-Tuning, Low Overhead Replacement Cache”.
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
