# Harness `Op` mapping (shared)

Standard mapping from harness [`Op<K>`](../../../../tests/abstract_models/mod.rs) to cache traits. Copy into policy specs; adjust side effects per policy.

| `Op` | Cache API | Typical side effects |
|------|-----------|----------------------|
| `Insert(k)` | `insert(k, v)` | Evict if full; may promote on re-insert (policy-specific) |
| `Get(k)` | `get(k)` | Promote on hit (recency / frequency policies) |
| `Peek(k)` | `peek(k)` | **No promotion** on LRU-family policies |
| `GetMut(k)` | `get_mut(k)` | Promote on hit where adapter models it |
| `Touch(k)` | `touch(k)` | Promote on hit |
| `Remove(k)` | `remove(k)` | Remove key; ordering side effects policy-specific |
| `EvictOne` | `evict_one()` | Evict victim per policy rule |

Align with [trait hierarchy](../../../design/trait-hierarchy.md): `Peek` must not change `recency_rank` on LRU-family policies.
