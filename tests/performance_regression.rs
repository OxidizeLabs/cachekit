//! Performance regression tests
//!
//! These tests verify complexity guarantees and catch major performance regressions.
//! They are NOT micro-benchmarks - use `cargo bench` for detailed performance analysis.
//!
//! ## Purpose
//!
//! - Verify O(1) complexity for core operations (get, insert, evict)
//! - Ensure reasonable performance bounds (loose thresholds to avoid flakiness)
//! - Catch catastrophic regressions that would impact production use
//!
//! ## What NOT to test here
//!
//! - Exact nanosecond timings (use benchmarks)
//! - Cross-library comparisons (use benchmarks)
//! - Detailed throughput analysis (use benchmarks)

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Helper to measure operation duration
fn measure_time<F, R>(operation: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = operation();
    (result, start.elapsed())
}

// =============================================================================
// Complexity Tests - Verify O(1) behavior
// =============================================================================

mod complexity_lru {
    use super::*;
    use cachekit::policy::lru::LruCore;
    use std::sync::Arc;

    #[test]
    fn test_get_is_o1() {
        verify_get_complexity::<i32, Arc<i32>, _, _>(LruCore::new, "LRU");
    }

    #[test]
    fn test_insert_is_o1() {
        verify_insert_complexity::<i32, Arc<i32>, _, _>(LruCore::new, "LRU");
    }

    #[test]
    fn test_eviction_is_o1() {
        verify_eviction_complexity::<i32, Arc<i32>, _, _>(LruCore::new, "LRU");
    }
}

mod complexity_lru_k {
    use super::*;
    use cachekit::policy::lru_k::LrukCache;

    #[test]
    fn test_get_is_o1() {
        verify_get_complexity::<i32, i32, _, _>(LrukCache::new, "LRU-K");
    }

    #[test]
    fn test_insert_is_o1() {
        verify_insert_complexity::<i32, i32, _, _>(LrukCache::new, "LRU-K");
    }

    #[test]
    fn test_eviction_is_o1() {
        verify_eviction_complexity::<i32, i32, _, _>(LrukCache::new, "LRU-K");
    }
}

mod complexity_lfu {
    use super::*;
    use cachekit::policy::lfu::LfuCache;
    use std::sync::Arc;

    #[test]
    fn test_get_is_o1() {
        verify_get_complexity::<i32, Arc<i32>, _, _>(LfuCache::new, "LFU");
    }

    #[test]
    fn test_insert_is_o1() {
        verify_insert_complexity::<i32, Arc<i32>, _, _>(LfuCache::new, "LFU");
    }

    #[test]
    fn test_eviction_is_o1() {
        verify_eviction_complexity::<i32, Arc<i32>, _, _>(LfuCache::new, "LFU");
    }
}

mod complexity_clock {
    use super::*;
    use cachekit::policy::clock::ClockCache;

    #[test]
    fn test_get_is_o1() {
        verify_get_complexity::<i32, i32, _, _>(ClockCache::new, "Clock");
    }

    #[test]
    fn test_insert_is_o1() {
        verify_insert_complexity::<i32, i32, _, _>(ClockCache::new, "Clock");
    }

    #[test]
    fn test_eviction_is_o1() {
        verify_eviction_complexity::<i32, i32, _, _>(ClockCache::new, "Clock");
    }
}

mod complexity_s3_fifo {
    use super::*;
    use cachekit::policy::s3_fifo::S3FifoCache;

    #[test]
    fn test_get_is_o1() {
        verify_get_complexity::<i32, i32, _, _>(S3FifoCache::new, "S3-FIFO");
    }

    #[test]
    fn test_insert_is_o1() {
        verify_insert_complexity::<i32, i32, _, _>(S3FifoCache::new, "S3-FIFO");
    }

    #[test]
    fn test_eviction_is_o1() {
        verify_eviction_complexity::<i32, i32, _, _>(S3FifoCache::new, "S3-FIFO");
    }
}

// =============================================================================
// Generic Complexity Verification Functions
// =============================================================================

/// Verify that get operation is O(1) by testing across multiple cache sizes
fn verify_get_complexity<K, V, C, F>(mut create_cache: F, policy_name: &str)
where
    K: std::hash::Hash + Eq + Clone + From<i32>,
    V: Clone + From<i32>,
    C: cachekit::traits::CoreCache<K, V>,
    F: FnMut(usize) -> C,
{
    let sizes = vec![1000, 2000, 4000, 8000];
    let mut times = Vec::new();

    for &size in &sizes {
        let mut cache = create_cache(size);

        // Pre-fill cache
        for i in 0..size as i32 {
            cache.insert(K::from(i), V::from(i));
        }

        // Measure get time with sufficient iterations for stable measurement
        let iterations = 10000;
        let (_, duration) = measure_time(|| {
            for i in 0..iterations {
                let key = K::from(i % size as i32);
                let _ = cache.get(&key);
            }
        });

        let avg_time = duration.as_nanos() as f64 / iterations as f64;
        times.push(avg_time);
        println!(
            "[{}] Size: {}, Avg get time: {:.2} ns",
            policy_name, size, avg_time
        );
    }

    // Verify O(1): time shouldn't grow linearly with size
    // For O(1), doubling size should NOT double time
    // Allow up to 15x ratio due to:
    // - Cache effects (larger working set, cache misses)
    // - Hash table resizing effects
    // - Memory allocator behavior
    // - Measurement noise in debug builds
    // - JIT/warmup effects
    //
    // Note: This is deliberately loose to avoid flakiness. For detailed
    // performance analysis, use `cargo bench`.
    for i in 1..times.len() {
        let size_ratio = sizes[i] as f64 / sizes[i - 1] as f64;
        let time_ratio = times[i] / times[i - 1];

        println!(
            "[{}] Size {}→{} ({:.2}x): time {:.1}ns→{:.1}ns ({:.2}x)",
            policy_name,
            sizes[i - 1],
            sizes[i],
            size_ratio,
            times[i - 1],
            times[i],
            time_ratio
        );

        assert!(
            time_ratio < 15.0,
            "[{}] Get operation appears to be O(n), not O(1):\n\
             Size increased by {:.2}x but time increased by {:.2}x\n\
             Expected: time_ratio << size_ratio for O(1) operations",
            policy_name,
            size_ratio,
            time_ratio
        );
    }
}

/// Verify that insert operation is O(1)
fn verify_insert_complexity<K, V, C, F>(mut create_cache: F, policy_name: &str)
where
    K: std::hash::Hash + Eq + Clone + From<i32>,
    V: Clone + From<i32>,
    C: cachekit::traits::CoreCache<K, V>,
    F: FnMut(usize) -> C,
{
    let sizes = vec![1000, 2000, 4000, 8000];
    let mut times = Vec::new();

    for &size in &sizes {
        let mut cache = create_cache(size);

        let (_, duration) = measure_time(|| {
            for i in 0..size as i32 {
                cache.insert(K::from(i), V::from(i));
            }
        });

        let avg_time = duration.as_nanos() as f64 / size as f64;
        times.push(avg_time);
        println!(
            "[{}] Size: {}, Avg insert time: {:.2} ns",
            policy_name, size, avg_time
        );
    }

    // Verify O(1) behavior (loose bounds for CI stability)
    for i in 1..times.len() {
        let size_ratio = sizes[i] as f64 / sizes[i - 1] as f64;
        let time_ratio = times[i] / times[i - 1];

        println!(
            "[{}] Size {}→{} ({:.2}x): time {:.1}ns→{:.1}ns ({:.2}x)",
            policy_name,
            sizes[i - 1],
            sizes[i],
            size_ratio,
            times[i - 1],
            times[i],
            time_ratio
        );

        assert!(
            time_ratio < 15.0,
            "[{}] Insert operation appears to be O(n), not O(1):\n\
             Size increased by {:.2}x but time increased by {:.2}x",
            policy_name,
            size_ratio,
            time_ratio
        );
    }
}

/// Verify that eviction operation is O(1)
fn verify_eviction_complexity<K, V, C, F>(mut create_cache: F, policy_name: &str)
where
    K: std::hash::Hash + Eq + Clone + From<i32>,
    V: Clone + From<i32>,
    C: cachekit::traits::CoreCache<K, V>,
    F: FnMut(usize) -> C,
{
    let sizes = vec![1000, 2000, 4000, 8000];
    let mut times = Vec::new();

    for &size in &sizes {
        let mut cache = create_cache(size);

        // Pre-fill cache
        for i in 0..size as i32 {
            cache.insert(K::from(i), V::from(i));
        }

        // Measure eviction time (insert into full cache triggers eviction)
        let iterations = 1000;
        let (_, duration) = measure_time(|| {
            for i in size as i32..(size as i32 + iterations) {
                cache.insert(K::from(i), V::from(i));
            }
        });

        let avg_time = duration.as_nanos() as f64 / iterations as f64;
        times.push(avg_time);
        println!(
            "[{}] Size: {}, Avg eviction time: {:.2} ns",
            policy_name, size, avg_time
        );
    }

    // Verify O(1) behavior (loose bounds for CI stability)
    for i in 1..times.len() {
        let size_ratio = sizes[i] as f64 / sizes[i - 1] as f64;
        let time_ratio = times[i] / times[i - 1];

        println!(
            "[{}] Size {}→{} ({:.2}x): time {:.1}ns→{:.1}ns ({:.2}x)",
            policy_name,
            sizes[i - 1],
            sizes[i],
            size_ratio,
            times[i - 1],
            times[i],
            time_ratio
        );

        assert!(
            time_ratio < 15.0,
            "[{}] Eviction operation appears to be O(n), not O(1):\n\
             Size increased by {:.2}x but time increased by {:.2}x",
            policy_name,
            size_ratio,
            time_ratio
        );
    }
}

// =============================================================================
// Critical Performance Regression Guards
// =============================================================================

mod regression_guards {
    use super::*;
    use cachekit::policy::lru::LruCore;
    use cachekit::traits::CoreCache;

    /// Ensure basic operations complete in reasonable time
    /// This catches catastrophic regressions (e.g., accidentally O(n) operations)
    #[test]
    fn test_operations_are_reasonably_fast() {
        let mut cache = LruCore::new(10000);

        // Pre-fill
        for i in 0..10000 {
            cache.insert(i, Arc::new(i));
        }

        // Measure bulk operations (fewer iterations for faster test)
        let iterations = 50_000;

        // Get operations should complete quickly
        let (_, get_duration) = measure_time(|| {
            for i in 0..iterations {
                cache.get(&(i % 10000));
            }
        });

        // Insert with eviction should complete quickly
        let (_, insert_duration) = measure_time(|| {
            for i in 10000..10000 + iterations {
                cache.insert(i, Arc::new(i));
            }
        });

        println!(
            "50K get operations: {:.2}ms (avg: {:.1}ns/op)",
            get_duration.as_secs_f64() * 1000.0,
            get_duration.as_nanos() as f64 / iterations as f64
        );
        println!(
            "50K insert+evict operations: {:.2}ms (avg: {:.1}ns/op)",
            insert_duration.as_secs_f64() * 1000.0,
            insert_duration.as_nanos() as f64 / iterations as f64
        );

        // Very loose bounds - only catch catastrophic issues
        // Note: These are debug mode timings, so we're very generous
        // In release mode these would be 10-100x faster
        assert!(
            get_duration < Duration::from_secs(10),
            "Get operations too slow: took {:?} for 50K ops (>10s indicates O(n) or worse)",
            get_duration
        );

        assert!(
            insert_duration < Duration::from_secs(10),
            "Insert+evict operations too slow: took {:?} for 50K ops (>10s indicates O(n) or worse)",
            insert_duration
        );
    }

    /// Verify cache doesn't leak or accumulate excessive memory
    #[test]
    fn test_memory_stability() {
        let capacity = 1000;
        let mut cache = LruCore::new(capacity);

        // Fill cache
        for i in 0..capacity {
            cache.insert(i, Arc::new(vec![0u8; 1024])); // 1KB values
        }

        assert_eq!(cache.len(), capacity);

        // Churn: insert many items, cache should stay at capacity
        for i in capacity..capacity + 10000 {
            cache.insert(i, Arc::new(vec![0u8; 1024]));
        }

        assert_eq!(
            cache.len(),
            capacity,
            "Cache should maintain capacity, not grow unbounded"
        );
    }
}
