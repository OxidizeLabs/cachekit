# CAR (Clock with Adaptive Replacement)

## Goal
ARC-like adaptivity with Clock mechanics to reduce list manipulation overhead.

## Core Idea
Replace ARC’s LRU lists with Clock structures plus ghost history:
- Maintain clock hands for resident sets and ghost sets.
- Use reference bits to approximate recency within each set.
- Use ARC-like feedback from ghost hits to adjust the balance parameter.

## Core Data Structures (Typical)
- `HashMap<K, Meta>` for list/clock membership and slot location
- Two Clock rings for resident sets (analogous to `T1`/`T2`)
- Two ghost structures (keys only) for history (analogous to `B1`/`B2`)

## Notes
CAR is not “one canonical implementation”; correctness and performance depend on details:
- how reference bits are set/cleared
- how ghost hits translate into tuning adjustments
- how replacement selects victims between the two rings

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
