# Intrusive Doubly-Linked List (O(1) Splice)

## What It Is
An **intrusive list** stores `prev/next` pointers *inside* the node itself. This makes list operations O(1) without allocating wrapper list nodes.

Used for:
- LRU / MRU ordering (move-to-front on hit)
- Segmented LRU (probation/protected lists)
- ARC/2Q lists (multiple LRU lists)
- “LRU within a bucket” tie-breakers (LFU bucket lists)

## Core Types

You need:
- `Node { prev: Option<Handle>, next: Option<Handle>, ... }`
- `List { head: Option<Handle>, tail: Option<Handle>, len: usize }`

Where `Handle` is one of:
- `NonNull<Node>` (pointer-based)
- `EntryId`/`usize` index into a stable arena/slot vector (recommended)

## Operations (Must Be O(1))
- `push_front(handle)`
- `push_back(handle)`
- `remove(handle)`
- `move_to_front(handle)` (remove + push_front)
- `pop_front()` / `pop_back()`

## Invariants
Maintain these at all times:
- `head.prev == None`
- `tail.next == None`
- For any node `n` in the list:
  - if `n.next = Some(m)`, then `m.prev = Some(n)`
  - if `n.prev = Some(p)`, then `p.next = Some(n)`
- `len` matches number of nodes reachable from head (debug-only validation is useful)

## Rust Implementation Notes

### Pointer-based (`NonNull<Node>`)
Pros: direct, classic LRU design. Cons: unsafe invariants and lifetime management.
- Nodes must have **stable addresses** (`Box`, arena allocation).
- Never move nodes after linking.
- Always unlink before freeing.

### Index-based (`usize`/`EntryId`)
Pros: usually safe, cache-friendly, easier to test. Cons: requires a stable slot array.
- Store nodes in `Vec<NodeSlot>`; each node stores `prev/next` as indices.
- Removal is purely index manipulation; no raw pointers.

Index-based is a good “generalized DS” because it composes cleanly with `slot-arena.md`.

## Common Pitfalls
- Forgetting to update both sides of a link on remove.
- Double-removing a node (keep an “is_linked” flag or use `Option<Entry>` slots).
- Using `Vec` of nodes without stable slots (reallocation moves nodes; breaks pointer-based lists).

## Testing Tips
- Property-test sequences of operations: push/remove/move/pop.
- Validate invariants after every op in debug builds.
- Include adversarial patterns (remove head/tail/middle repeatedly).
