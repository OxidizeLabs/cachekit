# API Surface

This page maps the major modules and how they fit together.

## Core Modules

- `builder` — `CacheBuilder` and `CachePolicy`, the unified entrypoint for building caches.
- `policy` — Policy implementations (LRU, LFU, LRU-K, 2Q, Clock, S3-FIFO).
- `ds` — Data structures used by policies (rings, lists, heaps, arenas).
- `stores` — Storage backends for policy metadata and values.
- `traits` — Common cache traits for shared behavior and policy-specific operations.

## Typical Flow

1. Choose a `CachePolicy` variant.
2. Build with `CacheBuilder::new(capacity).build::<K, V>(policy)`.
3. Use the unified `Cache<K, V>` API for standard operations.
4. Drop into `policy::*` or `ds::*` for advanced or policy-specific operations.

## Where to Start

- [Builder overview](integration.md)
- [Policy guidance](policies/README.md)
- [Data structures](policy-ds/README.md)
