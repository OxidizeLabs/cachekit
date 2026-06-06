//! Shared helpers for integration test binaries.
#![allow(dead_code)]

use cachekit::builder::{CacheBuilder, CachePolicy, DynCache};

/// All `CachePolicy` variants enabled by the current feature set.
///
/// Mirrors the list in `src/builder.rs` unit tests. CAR is excluded (not wired
/// into `CachePolicy`).
pub fn all_enabled_policies() -> Vec<CachePolicy> {
    vec![
        #[cfg(feature = "policy-fifo")]
        CachePolicy::Fifo,
        #[cfg(feature = "policy-lru")]
        CachePolicy::Lru,
        #[cfg(feature = "policy-fast-lru")]
        CachePolicy::FastLru,
        #[cfg(feature = "policy-lru-k")]
        CachePolicy::LruK { k: 2 },
        #[cfg(feature = "policy-lfu")]
        CachePolicy::Lfu { bucket_hint: None },
        #[cfg(feature = "policy-heap-lfu")]
        CachePolicy::HeapLfu,
        #[cfg(feature = "policy-two-q")]
        CachePolicy::TwoQ {
            probation_frac: 0.25,
        },
        #[cfg(feature = "policy-s3-fifo")]
        CachePolicy::S3Fifo {
            small_ratio: 0.1,
            ghost_ratio: 0.9,
        },
        #[cfg(feature = "policy-arc")]
        CachePolicy::Arc,
        #[cfg(feature = "policy-lifo")]
        CachePolicy::Lifo,
        #[cfg(feature = "policy-mfu")]
        CachePolicy::Mfu,
        #[cfg(feature = "policy-mru")]
        CachePolicy::Mru,
        #[cfg(feature = "policy-random")]
        CachePolicy::Random,
        #[cfg(feature = "policy-slru")]
        CachePolicy::Slru {
            probationary_frac: 0.25,
        },
        #[cfg(feature = "policy-clock")]
        CachePolicy::Clock,
        #[cfg(feature = "policy-clock-pro")]
        CachePolicy::ClockPro,
        #[cfg(feature = "policy-nru")]
        CachePolicy::Nru,
    ]
}

fn sample_v(n: u64) -> String {
    format!("v{n}")
}

/// Smoke-test the unified `Cache` surface on a `DynCache`.
pub fn exercise_dyn_cache(cache: &mut DynCache<u64, String>) {
    assert_eq!(cache.insert(1, sample_v(1)), None);
    assert_eq!(cache.insert(2, sample_v(2)), None);

    assert_eq!(cache.get(&1), Some(&sample_v(1)));
    assert_eq!(cache.peek(&2), Some(&sample_v(2)));
    assert!(cache.contains(&1));
    assert!(!cache.contains(&99));
    assert_eq!(cache.len(), 2);

    assert_eq!(cache.insert(1, sample_v(10)), Some(sample_v(1)));
    assert_eq!(cache.get(&1), Some(&sample_v(10)));

    cache.clear();
    assert!(cache.is_empty());
}

/// Build a `DynCache` for each enabled policy and run [`exercise_dyn_cache`].
pub fn exercise_all_dyn_caches(capacity: usize) {
    for policy in all_enabled_policies() {
        let mut cache: DynCache<u64, String> = CacheBuilder::new(capacity).build(policy);
        exercise_dyn_cache(&mut cache);
    }
}
