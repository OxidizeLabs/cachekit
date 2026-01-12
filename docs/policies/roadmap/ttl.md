# TTL / Time-Based Expiration

TTL is not a replacement policy; it’s an expiration rule that often coexists with an eviction policy.

## Implementation Patterns

### 1) Lazy expiration on access
Store `expires_at` per entry.
- On `get`: if expired, remove and treat as miss.
- On `insert`: set `expires_at = now + ttl`.

Pros: no background work. Cons: expired entries can occupy space until touched.

### 2) Timer wheel / min-heap expiry
Maintain an expiration index:
- min-heap keyed by `expires_at` (lazy stale entries), or
- timer wheel buckets for O(1) amortized expiry

Pros: can proactively free space. Cons: extra metadata and background/maintenance work.

## Interaction With Eviction
When cache is full:
- Prefer evicting expired entries first (cheap win).
- Then fall back to your policy (LRU/LFU/etc).

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
