//! Integration tests for `CacheBuilder` / `DynCache` dispatch.

mod common;

use std::sync::Arc;

use cachekit::builder::{CacheBuilder, CachePolicy, DynCache};
use cachekit::policy::fifo::FifoCache;
use cachekit::policy::lru::LruCore;
use cachekit::policy::s3_fifo::S3FifoCache;
use cachekit::traits::Cache;

mod cross_policy_parity {
    use super::*;

    #[test]
    fn all_enabled_policies_exercise_cache() {
        common::exercise_all_dyn_caches(10);
    }
}

mod peek_recency_semantics {
    use super::*;

    #[test]
    #[cfg(feature = "policy-lru")]
    fn lru_peek_does_not_promote() {
        let mut cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::Lru);
        cache.insert(1, 1);
        cache.insert(2, 2);
        cache.insert(3, 3);
        assert!(cache.peek(&2).is_some());
        cache.insert(4, 4);
        assert!(!cache.contains(&1));
        assert!(cache.contains(&3));
    }

    #[test]
    #[cfg(feature = "policy-lru")]
    fn lru_get_promotes_recency() {
        let mut cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::Lru);
        cache.insert(1, 1);
        cache.insert(2, 2);
        cache.insert(3, 3);
        assert!(cache.get(&1).is_some());
        assert!(cache.get(&2).is_some());
        cache.insert(4, 4);
        assert!(!cache.contains(&3));
        assert!(cache.contains(&1));
    }

    #[test]
    #[cfg(feature = "policy-fast-lru")]
    fn fast_lru_peek_vs_get() {
        let mut cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::FastLru);
        cache.insert(1, 1);
        cache.insert(2, 2);
        cache.insert(3, 3);
        assert!(cache.peek(&2).is_some());
        cache.insert(4, 4);
        assert!(!cache.contains(&1));

        let mut cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::FastLru);
        cache.insert(1, 1);
        cache.insert(2, 2);
        cache.insert(3, 3);
        assert!(cache.get(&1).is_some());
        assert!(cache.get(&2).is_some());
        cache.insert(4, 4);
        assert!(!cache.contains(&3));
    }

    #[test]
    #[cfg(feature = "policy-fifo")]
    fn fifo_recency_ops_are_no_ops() {
        let mut cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::Fifo);
        cache.insert(1, 1);
        cache.insert(2, 2);
        cache.insert(3, 3);
        assert!(cache.peek(&2).is_some());
        cache.insert(4, 4);
        assert!(!cache.contains(&1));

        let mut cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::Fifo);
        cache.insert(1, 1);
        cache.insert(2, 2);
        cache.insert(3, 3);
        assert!(cache.get(&2).is_some());
        cache.insert(4, 4);
        assert!(!cache.contains(&1));
    }
}

mod remove_and_update {
    use super::*;

    #[test]
    #[cfg(feature = "policy-lru")]
    fn lru_remove_returns_owned_value() {
        let mut cache = CacheBuilder::new(4).build::<u64, String>(CachePolicy::Lru);
        cache.insert(1, "one".to_string());
        assert_eq!(cache.remove(&1), Some("one".to_string()));
        assert!(!cache.contains(&1));
    }

    #[test]
    #[cfg(feature = "policy-lfu")]
    fn lfu_update_returns_previous() {
        let mut cache =
            CacheBuilder::new(4).build::<u64, String>(CachePolicy::Lfu { bucket_hint: None });
        cache.insert(1, "one".to_string());
        assert_eq!(cache.insert(1, "ONE".to_string()), Some("one".to_string()));
    }

    #[test]
    #[cfg(feature = "policy-heap-lfu")]
    fn heap_lfu_update_returns_previous() {
        let mut cache = CacheBuilder::new(4).build::<u64, String>(CachePolicy::HeapLfu);
        cache.insert(1, "one".to_string());
        assert_eq!(cache.insert(1, "ONE".to_string()), Some("one".to_string()));
    }
}

mod validation_panics {
    use super::*;

    #[test]
    #[cfg(feature = "policy-lru")]
    #[should_panic(expected = "cache capacity must be greater than 0")]
    fn builder_zero_capacity_panics() {
        let _ = CacheBuilder::new(0).build::<u64, String>(CachePolicy::Lru);
    }

    #[test]
    #[cfg(feature = "policy-lru-k")]
    #[should_panic(expected = "LruK: k must be greater than 0")]
    fn lru_k_zero_k_panics() {
        let _ = CacheBuilder::new(10).build::<u64, String>(CachePolicy::LruK { k: 0 });
    }

    #[test]
    #[cfg(feature = "policy-two-q")]
    #[should_panic(expected = "TwoQ: probation_frac must be a finite value in 0.0..=1.0")]
    fn two_q_invalid_frac_panics() {
        let _ = CacheBuilder::new(10).build::<u64, String>(CachePolicy::TwoQ {
            probation_frac: 1.5,
        });
    }

    #[test]
    #[cfg(feature = "policy-s3-fifo")]
    #[should_panic(expected = "S3Fifo: small_ratio must be a finite value in 0.0..=1.0")]
    fn s3_fifo_invalid_small_ratio_panics() {
        let _ = CacheBuilder::new(10).build::<u64, String>(CachePolicy::S3Fifo {
            small_ratio: 2.0,
            ghost_ratio: 0.5,
        });
    }

    #[test]
    #[cfg(feature = "policy-slru")]
    #[should_panic(expected = "Slru: probationary_frac must be a finite value in 0.0..=1.0")]
    fn slru_invalid_frac_panics() {
        let _ = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Slru {
            probationary_frac: -0.1,
        });
    }
}

mod config_propagation {
    use super::*;

    #[test]
    #[cfg(feature = "policy-lfu")]
    fn lfu_bucket_hint_builds() {
        let mut cache = CacheBuilder::new(8).build::<u64, String>(CachePolicy::Lfu {
            bucket_hint: Some(64),
        });
        cache.insert(1, "x".to_string());
        assert!(cache.contains(&1));
    }

    #[test]
    #[cfg(feature = "policy-two-q")]
    fn two_q_boundary_fraction_builds() {
        let mut cache = CacheBuilder::new(8).build::<u64, String>(CachePolicy::TwoQ {
            probation_frac: 0.0,
        });
        cache.insert(1, "x".to_string());
        assert_eq!(cache.len(), 1);
    }
}

mod trait_dispatch {
    use super::*;

    #[test]
    #[cfg(feature = "policy-fifo")]
    fn dyn_cache_via_cache_trait() {
        let mut cache: DynCache<u64, String> = CacheBuilder::new(8).build(CachePolicy::Fifo);
        common::exercise_dyn_cache(&mut cache);
    }
}

mod debug_output {
    use super::*;

    #[test]
    #[cfg(feature = "policy-lru")]
    fn debug_contains_policy_name_and_len() {
        let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);
        cache.insert(1, "one".to_string());
        let debug = format!("{:?}", cache);
        assert!(debug.contains("DynCache"));
        assert!(debug.contains("Lru"));
        assert!(debug.contains("len: 1"));
    }
}

mod runtime_selection {
    use super::*;

    fn build_from_name(name: &str) -> DynCache<u64, String> {
        let policy = match name {
            #[cfg(feature = "policy-lru")]
            "lru" => CachePolicy::Lru,
            #[cfg(feature = "policy-fifo")]
            "fifo" => CachePolicy::Fifo,
            #[cfg(feature = "policy-fast-lru")]
            "fast_lru" => CachePolicy::FastLru,
            _ => panic!("unknown policy {name}"),
        };
        CacheBuilder::new(8).build(policy)
    }

    #[test]
    fn config_driven_dispatch_smoke() {
        #[cfg(feature = "policy-lru")]
        {
            let mut cache = build_from_name("lru");
            cache.insert(1, "a".to_string());
            assert!(cache.contains(&1));
        }
        #[cfg(feature = "policy-fifo")]
        {
            let mut cache = build_from_name("fifo");
            cache.insert(2, "b".to_string());
            assert!(cache.contains(&2));
        }
    }
}

mod equivalence_smoke {
    use super::*;

    #[test]
    #[cfg(feature = "policy-lru")]
    fn lru_dyn_matches_direct_core() {
        let mut dyn_cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::Lru);
        let mut direct = LruCore::<u64, u64>::new(3);

        for key in [10u64, 20, 30] {
            dyn_cache.insert(key, key);
            direct.insert(key, Arc::new(key));
        }
        assert_eq!(dyn_cache.peek(&20).copied(), direct.peek(&20).map(|v| *v));
        dyn_cache.insert(40, 40);
        direct.insert(40, Arc::new(40));

        for key in [20u64, 30, 40] {
            assert_eq!(dyn_cache.contains(&key), direct.contains(&key));
        }
        assert_eq!(dyn_cache.len(), direct.len());
    }

    #[test]
    #[cfg(feature = "policy-fifo")]
    fn fifo_dyn_matches_direct() {
        let mut dyn_cache = CacheBuilder::new(3).build::<u64, u64>(CachePolicy::Fifo);
        let mut direct = FifoCache::<u64, u64>::new(3);

        for key in [1u64, 2, 3] {
            dyn_cache.insert(key, key);
            direct.insert(key, key);
        }
        dyn_cache.insert(4, 4);
        direct.insert(4, 4);

        assert_eq!(dyn_cache.len(), direct.len());
        assert_eq!(dyn_cache.contains(&1), direct.contains(&1));
        assert_eq!(dyn_cache.contains(&4), direct.contains(&4));
    }

    #[test]
    #[cfg(feature = "policy-s3-fifo")]
    fn s3_fifo_dyn_matches_direct() {
        let mut dyn_cache = CacheBuilder::new(10).build::<u64, u64>(CachePolicy::S3Fifo {
            small_ratio: 0.1,
            ghost_ratio: 0.9,
        });
        let mut direct = S3FifoCache::<u64, u64>::new(10);

        for key in 0..8u64 {
            dyn_cache.insert(key, key);
            direct.insert(key, key);
        }
        assert_eq!(dyn_cache.len(), direct.len());
        for key in 0..8u64 {
            assert_eq!(dyn_cache.contains(&key), direct.contains(&key));
        }
    }
}

#[cfg(feature = "policy-all")]
mod drift_guard {
    use super::common;

    #[test]
    fn all_enabled_policies_matches_builder_wiring() {
        assert_eq!(common::all_enabled_policies().len(), 17);
    }
}
