# Stores

CacheKit “stores” are the underlying key/value containers used by policies. They provide:

- A capacity limit (by entries, weight, or other accounting)
- Basic operations (get/insert/remove/clear)
- Optional concurrency wrappers
- Metrics counters (hits/misses/inserts/updates/removes/evictions)

Most policies are generic over store traits defined in `cachekit::store::traits`:

- `StoreCore<K, V>`: read-only operations + metrics
- `StoreMut<K, V>`: mutation operations
- `ConcurrentStore<K, V>`: `Send + Sync` stores (typically via `RwLock`)

## Store implementations

- [HashMap store](hashmap.md): simplest entry-count store; has concurrent and sharded variants.
- [Slab store](slab.md): stable `EntryId` handles via indirection; good for policy metadata.
- [Weight store](weight.md): enforces both entry count and total “weight” (e.g. bytes).
- [Handle store](handle.md): keyed by compact handles (IDs) instead of full keys.

## Choosing a store (quick guide)

- Default: use the HashMap store.
- If the policy needs stable entry IDs: use the Slab store.
- If you want size-based capacity: use the Weight store.
- If you already have an interner / stable IDs for keys: use the Handle store.
