# Slot Arena (Stable Handles + Free List)

## What It Is
A **slot arena** stores entries in a `Vec<Option<T>>` (or equivalent) and returns stable handles (indices) that remain valid until freed.

Used for:
- avoiding pointer-based intrusive lists
- storing policy metadata densely (good cache locality)
- fast allocation/deallocation via a free list

This pattern appears in `cachekit` policies (LFU/LRU-K) and in `SlabStore` (`src/store/slab.rs`).

## Core Types
- `EntryId(usize)` (opaque handle)
- `slots: Vec<Option<T>>`
- `free_list: Vec<usize>`

## Allocation
`allocate(value) -> EntryId`
- If `free_list` not empty: reuse that index.
- Else: `slots.push(None)` then fill the new slot.

## Removal
`remove(id) -> T`
- `take()` the `Option<T>` from `slots[id]`
- push `id` into `free_list`

## Invariants
- A handle is valid if and only if `slots[id].is_some()`.
- Reusing freed indices must not leave old `prev/next` links in place (clear metadata on free).

## Why It Helps Policies
With a slot arena you can implement:
- intrusive lists using indices instead of pointers (`intrusive-list.md`)
- “bucket lists” for LFU (`frequency-buckets.md`)
- segmented lists (cold/hot, probation/protected) without allocations

## Rust Notes
- Prefer an opaque `EntryId` newtype over raw `usize`.
- Consider generation counters (slotmap-style) if you need to detect use-after-free handles.
- Keep “hot” metadata (prev/next/freq/segment flags) in a compact struct for cache efficiency.
