// ==============================================
// CROSS-POLICY INVARIANT TESTS (integration)
// ==============================================
//
// Tests that verify library-wide behavioral consistency across all cache
// policies. These span multiple modules and belong here rather than in any
// single source file.

// ==============================================
// Capacity-0 Behavior
// ==============================================
//
// Some policies silently coerce capacity=0 to capacity=1 via .max(1) in
// their constructor, which is inconsistent with the rest of the library.

#[cfg(feature = "policy-clock")]
mod clock_zero_capacity {
    use cachekit::policy::clock::ClockCache;
    use cachekit::traits::ReadOnlyCache;

    #[test]
    fn capacity_zero_is_honored() {
        let cache: ClockCache<&str, i32> = ClockCache::new(0);

        assert_eq!(
            cache.capacity(),
            0,
            "ClockCache::new(0) should honor capacity=0, not coerce to {}",
            cache.capacity()
        );
    }
}

#[cfg(feature = "policy-clock-pro")]
mod clock_pro_zero_capacity {
    use cachekit::policy::clock_pro::ClockProCache;
    use cachekit::traits::ReadOnlyCache;

    #[test]
    fn capacity_zero_is_honored() {
        let cache: ClockProCache<&str, i32> = ClockProCache::new(0);

        assert_eq!(
            cache.capacity(),
            0,
            "ClockProCache::new(0) should honor capacity=0, not coerce to {}",
            cache.capacity()
        );
    }

    #[test]
    fn capacity_zero_rejects_inserts() {
        use cachekit::traits::CoreCache;

        let mut cache: ClockProCache<&str, i32> = ClockProCache::new(0);
        cache.insert("key", 42);

        assert_eq!(
            cache.len(),
            0,
            "ClockProCache with capacity=0 should reject inserts"
        );
    }
}

#[cfg(feature = "policy-nru")]
mod nru_zero_capacity {
    use cachekit::policy::nru::NruCache;
    use cachekit::traits::ReadOnlyCache;

    #[test]
    fn capacity_zero_is_honored() {
        let cache: NruCache<&str, i32> = NruCache::new(0);

        assert_eq!(
            cache.capacity(),
            0,
            "NruCache::new(0) should honor capacity=0, not coerce to {}",
            cache.capacity()
        );
    }
}
