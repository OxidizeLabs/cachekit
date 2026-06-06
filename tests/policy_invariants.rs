// ==============================================
// CROSS-POLICY INVARIANT TESTS (integration)
// ==============================================

#[cfg(feature = "policy-fifo")]
mod fifo_zero_capacity {
    use cachekit::policy::fifo::FifoCache;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        let cache: FifoCache<&str, i32> = FifoCache::new(0);
        assert_eq!(cache.capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache: FifoCache<&str, i32> = FifoCache::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn try_new_zero_returns_err() {
        assert!(FifoCache::<&str, i32>::try_new(0).is_err());
    }
}

#[cfg(feature = "policy-lru")]
mod lru_zero_capacity {
    use std::sync::Arc;

    use cachekit::policy::lru::LruCore;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        let cache: LruCore<&str, i32> = LruCore::new(0);
        assert_eq!(cache.capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache: LruCore<&str, i32> = LruCore::new(0);
        cache.insert("key", Arc::new(42));
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-fast-lru")]
mod fast_lru_zero_capacity {
    use cachekit::policy::fast_lru::FastLru;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(FastLru::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = FastLru::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-lru-k")]
mod lru_k_zero_capacity {
    use cachekit::policy::lru_k::LrukCache;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(LrukCache::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = LrukCache::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-lfu")]
mod lfu_zero_capacity {
    use std::sync::Arc;

    use cachekit::policy::lfu::LfuCache;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(LfuCache::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = LfuCache::<&str, i32>::new(0);
        cache.insert("key", Arc::new(42));
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-heap-lfu")]
mod heap_lfu_zero_capacity {
    use std::sync::Arc;

    use cachekit::policy::heap_lfu::HeapLfuCache;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(HeapLfuCache::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = HeapLfuCache::<&str, i32>::new(0);
        cache.insert("key", Arc::new(42));
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-two-q")]
mod two_q_zero_capacity {
    use cachekit::policy::two_q::TwoQCore;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(TwoQCore::<&str, i32>::new(0, 0.25).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = TwoQCore::<&str, i32>::new(0, 0.25);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-s3-fifo")]
mod s3_fifo_zero_capacity {
    use cachekit::policy::s3_fifo::S3FifoCache;

    #[test]
    #[should_panic(expected = "cache capacity must be greater than zero")]
    fn new_zero_panics() {
        let _cache: S3FifoCache<&str, i32> = S3FifoCache::new(0);
    }

    #[test]
    fn try_with_ratios_zero_returns_err() {
        assert!(S3FifoCache::<&str, i32>::try_with_ratios(0, 0.1, 0.9).is_err());
    }
}

#[cfg(feature = "policy-arc")]
mod arc_zero_capacity {
    use cachekit::policy::arc::ArcCore;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(ArcCore::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = ArcCore::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-car")]
mod car_zero_capacity {
    use cachekit::policy::car::CarCore;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(CarCore::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = CarCore::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-lifo")]
mod lifo_zero_capacity {
    use cachekit::policy::lifo::LifoCore;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(LifoCore::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = LifoCore::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-mfu")]
mod mfu_zero_capacity {
    use cachekit::policy::mfu::MfuCore;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(MfuCore::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = MfuCore::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-mru")]
mod mru_zero_capacity {
    use cachekit::policy::mru::MruCore;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(MruCore::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = MruCore::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-random")]
mod random_zero_capacity {
    use cachekit::policy::random::RandomCore;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(RandomCore::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = RandomCore::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-slru")]
mod slru_zero_capacity {
    use cachekit::policy::slru::SlruCore;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(SlruCore::<&str, i32>::new(0, 0.25).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = SlruCore::<&str, i32>::new(0, 0.25);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-clock")]
mod clock_zero_capacity {
    use cachekit::policy::clock::ClockCache;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(ClockCache::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = ClockCache::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-clock-pro")]
mod clock_pro_zero_capacity {
    use cachekit::policy::clock_pro::ClockProCache;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(ClockProCache::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = ClockProCache::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

#[cfg(feature = "policy-nru")]
mod nru_zero_capacity {
    use cachekit::policy::nru::NruCache;
    use cachekit::traits::Cache;

    #[test]
    fn capacity_zero_is_honored() {
        assert_eq!(NruCache::<&str, i32>::new(0).capacity(), 0);
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        let mut cache = NruCache::<&str, i32>::new(0);
        cache.insert("key", 42);
        assert_eq!(cache.len(), 0);
    }
}

mod len_bounds {
    use cachekit::traits::Cache;

    #[test]
    #[cfg(feature = "policy-lru")]
    fn lru_len_bounded_after_churn() {
        use std::sync::Arc;

        use cachekit::policy::lru::LruCore;

        let capacity = 8;
        let mut cache = LruCore::<u64, i32>::new(capacity);
        for i in 0..(capacity * 2) as u64 {
            cache.insert(i, Arc::new(i as i32));
            assert!(cache.len() <= capacity);
        }
    }

    #[test]
    #[cfg(feature = "policy-s3-fifo")]
    fn s3_fifo_queue_lens_bounded_after_churn() {
        use cachekit::policy::s3_fifo::S3FifoCache;

        let capacity = 10;
        let mut cache = S3FifoCache::<u64, i32>::new(capacity);
        for i in 0..(capacity * 2) as u64 {
            cache.insert(i, i as i32);
            assert!(cache.small_len() + cache.main_len() <= capacity);
        }
    }
}
