# Clock Ring (Second-Chance)

## What It Is
The **Clock** policy maintains a circular set of slots and a moving “hand”. Each slot has a reference bit.

This is a general-purpose DS for:
- Clock / second-chance eviction
- CAR/CLOCK-Pro style variants (with extra status bits)

## Core Types
- `slots: Vec<SlotMeta>` (dense)
- `hand: usize`
- `index: HashMap<K, usize>` (key → slot)

Minimal `SlotMeta`:
- `key: K` (or empty)
- `referenced: bool`
- `occupied: bool`

## Operations

### `touch(key)`
- Find slot via index; set `referenced = true`.

### `insert(key, value)`
- If key exists: update value + `referenced = true`.
- Else: find a free slot or evict:
  - Loop: if `referenced` set → clear it and advance; else evict this slot.

### `evict_one() -> K`
- Advance hand until a victim is found; return victim key.

## Complexity
- O(1) amortized for eviction; worst-case can scan many slots if all are referenced.

## Implementation Notes
- Use a dense array of slots for locality; avoid linked lists.
- For predictable performance, consider limiting scan length or adding an epoch-based reference reset (NRU-like).
