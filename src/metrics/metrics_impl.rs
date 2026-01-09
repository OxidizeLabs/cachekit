use crate::metrics::cell::MetricsCell;

#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub stale_skips: u64,
    pub evict_scan_steps: u64,
    pub pop_oldest_calls: u64,
    pub pop_oldest_found: u64,
    pub pop_oldest_empty_or_stale: u64,
    pub peek_oldest_calls: MetricsCell,
    pub peek_oldest_found: MetricsCell,
    pub age_rank_calls: MetricsCell,
    pub age_rank_scan_steps: MetricsCell,
    pub age_rank_found: MetricsCell,
}

impl CacheMetrics {
    pub fn new() -> CacheMetrics {
        Self {
            get_calls: 0,
            get_hits: 0,
            get_misses: 0,
            insert_calls: 0,
            insert_updates: 0,
            insert_new: 0,
            evict_calls: 0,
            evicted_entries: 0,
            stale_skips: 0,
            evict_scan_steps: 0,
            pop_oldest_calls: 0,
            pop_oldest_found: 0,
            pop_oldest_empty_or_stale: 0,
            peek_oldest_calls: MetricsCell::new(),
            peek_oldest_found: MetricsCell::new(),
            age_rank_calls: MetricsCell::new(),
            age_rank_scan_steps: MetricsCell::new(),
            age_rank_found: MetricsCell::new(),
        }
    }
}
