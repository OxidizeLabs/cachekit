# FAQ and Gotchas

## Are caches thread-safe by default?

No. Wrap a cache in a lock (`Mutex`, `RwLock`) or enable the `concurrency` feature
to use built-in concurrent wrappers like `ConcurrentClockRing`.

## Does `get` update policy metadata?

Yes. `get` updates recency or frequency depending on the policy. Use `contains`
when you want to check presence without updating metadata.

## What happens on insert at capacity?

The cache evicts an entry according to the selected policy.

## Can I use zero capacity?

Policies handle zero capacity differently; some clamp to 1. Prefer explicit
capacity >= 1 unless you are validating edge cases.

## Why do some policies use `Arc<V>` internally?

Some policies store values behind `Arc` to minimize copies. The builder hides
this detail so the unified `Cache<K, V>` API still works with owned values.
