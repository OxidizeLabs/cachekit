# Frequency Buckets (O(1) LFU Core)

## What It Is
Bucketed LFU keeps entries grouped by frequency, with O(1) promotion and O(1) eviction using `min_freq`.

Generalized DS:
- A slot arena of entries (`EntryId`)
- For each `freq`, an intrusive list of entry ids (usually LRU within the bucket)
- A way to find the “current minimum non-empty frequency”

## Core Types
- `entries: Vec<Option<EntryMeta>>`
- `index: HashMap<K, EntryId>`
- `buckets: HashMap<u64, BucketMeta>`
- `min_freq: u64`

Where `BucketMeta` holds:
- `list_head/list_tail` (intrusive list of `EntryId`)
- optional `prev/next` links between buckets to skip empty buckets quickly

## Operations

### `touch(id)`
- Remove `id` from `freq` bucket list.
- Increment `freq`.
- Insert into new bucket list head.
- If old bucket empty and was `min_freq`, advance `min_freq`.

### `evict()`
- Choose victim from `min_freq` bucket tail (LRU tie-break).
- Remove from index + entry slots.

## Notes
- The hard part is **efficiently skipping empty frequencies**. Linking buckets (`prev/next`) is one approach; scanning is another (but can become O(n)).
- Be explicit about saturation/overflow behavior (`u64` counts can grow without bound; aging/decay is a separate concern).
