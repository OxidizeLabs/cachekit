#[derive(Debug, Default, Clone, Copy)]
pub struct CacheMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub stale_skips: u64, // queue entries popped that were already removed from map
    pub evict_scan_steps: u64, // how many pop_front iterations inside eviction

    pub pop_oldest_calls: u64,
    pub pop_oldest_found: u64,
    pub pop_oldest_empty_or_stale: u64,

    pub peek_oldest_calls: u64,
    pub peek_oldest_found: u64,

    pub age_rank_calls: u64,
    pub age_rank_found: u64,
    pub age_rank_scan_steps: u64,

    // gauges captured at snapshot time
    pub cache_len: usize,
    pub insertion_order_len: usize,
    pub capacity: usize,
}
