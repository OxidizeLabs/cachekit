use crate::metrics::cell::MetricsCell;
use crate::metrics::traits::{
    CoreMetricsRecorder, FifoMetricsReadRecorder, FifoMetricsRecorder, LfuMetricsReadRecorder,
    LfuMetricsRecorder, LruKMetricsReadRecorder, LruKMetricsRecorder, LruMetricsReadRecorder,
    LruMetricsRecorder,
};

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

#[derive(Debug, Default)]
pub struct LruMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub pop_lru_calls: u64,
    pub pop_lru_found: u64,
    pub peek_lru_calls: MetricsCell,
    pub peek_lru_found: MetricsCell,
    pub touch_calls: u64,
    pub touch_found: u64,
    pub recency_rank_calls: MetricsCell,
    pub recency_rank_found: MetricsCell,
    pub recency_rank_scan_steps: MetricsCell,
}

#[derive(Debug, Default)]
pub struct LfuMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub pop_lfu_calls: u64,
    pub pop_lfu_found: u64,
    pub peek_lfu_calls: MetricsCell,
    pub peek_lfu_found: MetricsCell,
    pub frequency_calls: MetricsCell,
    pub frequency_found: MetricsCell,
    pub reset_frequency_calls: u64,
    pub reset_frequency_found: u64,
    pub increment_frequency_calls: u64,
    pub increment_frequency_found: u64,
}

#[derive(Debug, Default)]
pub struct LruKMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub pop_lru_calls: u64,
    pub pop_lru_found: u64,
    pub peek_lru_calls: u64,
    pub peek_lru_found: u64,
    pub touch_calls: u64,
    pub touch_found: u64,
    pub recency_rank_calls: u64,
    pub recency_rank_found: u64,
    pub recency_rank_scan_steps: u64,
    pub pop_lru_k_calls: u64,
    pub pop_lru_k_found: u64,
    pub peek_lru_k_calls: MetricsCell,
    pub peek_lru_k_found: MetricsCell,
    pub k_distance_calls: MetricsCell,
    pub k_distance_found: MetricsCell,
    pub k_distance_rank_calls: MetricsCell,
    pub k_distance_rank_found: MetricsCell,
    pub k_distance_rank_scan_steps: MetricsCell,
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

impl CoreMetricsRecorder for CacheMetrics {
    fn record_get_hit(&mut self) {
        self.get_calls += 1;
        self.get_hits += 1;
    }

    fn record_get_miss(&mut self) {
        self.get_calls += 1;
        self.get_misses += 1;
    }

    fn record_insert_call(&mut self) {
        self.insert_calls += 1;
    }

    fn record_insert_new(&mut self) {
        self.insert_new += 1;
    }

    fn record_insert_update(&mut self) {
        self.insert_updates += 1;
    }

    fn record_evict_call(&mut self) {
        self.evict_calls += 1;
    }

    fn record_evicted_entry(&mut self) {
        self.evicted_entries += 1;
    }

    fn record_clear(&mut self) {
        // No explicit counter today; kept for trait completeness.
    }
}

impl FifoMetricsRecorder for CacheMetrics {
    fn record_evict_scan_step(&mut self) {
        self.evict_scan_steps += 1;
    }

    fn record_stale_skip(&mut self) {
        self.stale_skips += 1;
    }

    fn record_pop_oldest_call(&mut self) {
        self.pop_oldest_calls += 1;
    }

    fn record_pop_oldest_found(&mut self) {
        self.pop_oldest_found += 1;
    }

    fn record_pop_oldest_empty_or_stale(&mut self) {
        self.pop_oldest_empty_or_stale += 1;
    }
}

impl FifoMetricsReadRecorder for &CacheMetrics {
    fn record_peek_oldest_call(&self) {
        self.peek_oldest_calls.incr();
    }

    fn record_peek_oldest_found(&self) {
        self.peek_oldest_found.incr();
    }

    fn record_age_rank_call(&self) {
        self.age_rank_calls.incr();
    }

    fn record_age_rank_found(&self) {
        self.age_rank_found.incr();
    }

    fn record_age_rank_scan_step(&self) {
        self.age_rank_scan_steps.incr();
    }
}

impl CoreMetricsRecorder for LruMetrics {
    fn record_get_hit(&mut self) {
        self.get_calls += 1;
        self.get_hits += 1;
    }

    fn record_get_miss(&mut self) {
        self.get_calls += 1;
        self.get_misses += 1;
    }

    fn record_insert_call(&mut self) {
        self.insert_calls += 1;
    }

    fn record_insert_new(&mut self) {
        self.insert_new += 1;
    }

    fn record_insert_update(&mut self) {
        self.insert_updates += 1;
    }

    fn record_evict_call(&mut self) {
        self.evict_calls += 1;
    }

    fn record_evicted_entry(&mut self) {
        self.evicted_entries += 1;
    }

    fn record_clear(&mut self) {}
}

impl LruMetricsRecorder for LruMetrics {
    fn record_pop_lru_call(&mut self) {
        self.pop_lru_calls += 1;
    }

    fn record_pop_lru_found(&mut self) {
        self.pop_lru_found += 1;
    }

    fn record_peek_lru_call(&mut self) {
        self.peek_lru_calls.incr();
    }

    fn record_peek_lru_found(&mut self) {
        self.peek_lru_found.incr();
    }

    fn record_touch_call(&mut self) {
        self.touch_calls += 1;
    }

    fn record_touch_found(&mut self) {
        self.touch_found += 1;
    }

    fn record_recency_rank_call(&mut self) {
        self.recency_rank_calls.incr();
    }

    fn record_recency_rank_found(&mut self) {
        self.recency_rank_found.incr();
    }

    fn record_recency_rank_scan_step(&mut self) {
        self.recency_rank_scan_steps.incr();
    }
}

impl LruMetricsReadRecorder for &LruMetrics {
    fn record_peek_lru_call(&self) {
        self.peek_lru_calls.incr();
    }

    fn record_peek_lru_found(&self) {
        self.peek_lru_found.incr();
    }

    fn record_recency_rank_call(&self) {
        self.recency_rank_calls.incr();
    }

    fn record_recency_rank_found(&self) {
        self.recency_rank_found.incr();
    }

    fn record_recency_rank_scan_step(&self) {
        self.recency_rank_scan_steps.incr();
    }
}

impl CoreMetricsRecorder for LfuMetrics {
    fn record_get_hit(&mut self) {
        self.get_calls += 1;
        self.get_hits += 1;
    }

    fn record_get_miss(&mut self) {
        self.get_calls += 1;
        self.get_misses += 1;
    }

    fn record_insert_call(&mut self) {
        self.insert_calls += 1;
    }

    fn record_insert_new(&mut self) {
        self.insert_new += 1;
    }

    fn record_insert_update(&mut self) {
        self.insert_updates += 1;
    }

    fn record_evict_call(&mut self) {
        self.evict_calls += 1;
    }

    fn record_evicted_entry(&mut self) {
        self.evicted_entries += 1;
    }

    fn record_clear(&mut self) {}
}

impl LfuMetricsRecorder for LfuMetrics {
    fn record_pop_lfu_call(&mut self) {
        self.pop_lfu_calls += 1;
    }

    fn record_pop_lfu_found(&mut self) {
        self.pop_lfu_found += 1;
    }

    fn record_peek_lfu_call(&mut self) {
        self.peek_lfu_calls.incr();
    }

    fn record_peek_lfu_found(&mut self) {
        self.peek_lfu_found.incr();
    }

    fn record_frequency_call(&mut self) {
        self.frequency_calls.incr();
    }

    fn record_frequency_found(&mut self) {
        self.frequency_found.incr();
    }

    fn record_reset_frequency_call(&mut self) {
        self.reset_frequency_calls += 1;
    }

    fn record_reset_frequency_found(&mut self) {
        self.reset_frequency_found += 1;
    }

    fn record_increment_frequency_call(&mut self) {
        self.increment_frequency_calls += 1;
    }

    fn record_increment_frequency_found(&mut self) {
        self.increment_frequency_found += 1;
    }
}

impl LfuMetricsReadRecorder for &LfuMetrics {
    fn record_peek_lfu_call(&self) {
        self.peek_lfu_calls.incr();
    }

    fn record_peek_lfu_found(&self) {
        self.peek_lfu_found.incr();
    }

    fn record_frequency_call(&self) {
        self.frequency_calls.incr();
    }

    fn record_frequency_found(&self) {
        self.frequency_found.incr();
    }
}

impl CoreMetricsRecorder for LruKMetrics {
    fn record_get_hit(&mut self) {
        self.get_calls += 1;
        self.get_hits += 1;
    }

    fn record_get_miss(&mut self) {
        self.get_calls += 1;
        self.get_misses += 1;
    }

    fn record_insert_call(&mut self) {
        self.insert_calls += 1;
    }

    fn record_insert_new(&mut self) {
        self.insert_new += 1;
    }

    fn record_insert_update(&mut self) {
        self.insert_updates += 1;
    }

    fn record_evict_call(&mut self) {
        self.evict_calls += 1;
    }

    fn record_evicted_entry(&mut self) {
        self.evicted_entries += 1;
    }

    fn record_clear(&mut self) {}
}

impl LruMetricsRecorder for LruKMetrics {
    fn record_pop_lru_call(&mut self) {
        self.pop_lru_calls += 1;
    }

    fn record_pop_lru_found(&mut self) {
        self.pop_lru_found += 1;
    }

    fn record_peek_lru_call(&mut self) {
        self.peek_lru_calls += 1;
    }

    fn record_peek_lru_found(&mut self) {
        self.peek_lru_found += 1;
    }

    fn record_touch_call(&mut self) {
        self.touch_calls += 1;
    }

    fn record_touch_found(&mut self) {
        self.touch_found += 1;
    }

    fn record_recency_rank_call(&mut self) {
        self.recency_rank_calls += 1;
    }

    fn record_recency_rank_found(&mut self) {
        self.recency_rank_found += 1;
    }

    fn record_recency_rank_scan_step(&mut self) {
        self.recency_rank_scan_steps += 1;
    }
}

impl LruKMetricsRecorder for LruKMetrics {
    fn record_pop_lru_k_call(&mut self) {
        self.pop_lru_k_calls += 1;
    }

    fn record_pop_lru_k_found(&mut self) {
        self.pop_lru_k_found += 1;
    }

    fn record_peek_lru_k_call(&mut self) {
        self.peek_lru_k_calls.incr();
    }

    fn record_peek_lru_k_found(&mut self) {
        self.peek_lru_k_found.incr();
    }

    fn record_k_distance_call(&mut self) {
        self.k_distance_calls.incr();
    }

    fn record_k_distance_found(&mut self) {
        self.k_distance_found.incr();
    }

    fn record_k_distance_rank_call(&mut self) {
        self.k_distance_rank_calls.incr();
    }

    fn record_k_distance_rank_found(&mut self) {
        self.k_distance_rank_found.incr();
    }

    fn record_k_distance_rank_scan_step(&mut self) {
        self.k_distance_rank_scan_steps.incr();
    }
}

impl LruKMetricsReadRecorder for &LruKMetrics {
    fn record_peek_lru_k_call(&self) {
        self.peek_lru_k_calls.incr();
    }

    fn record_peek_lru_k_found(&self) {
        self.peek_lru_k_found.incr();
    }

    fn record_k_distance_call(&self) {
        self.k_distance_calls.incr();
    }

    fn record_k_distance_found(&self) {
        self.k_distance_found.incr();
    }

    fn record_k_distance_rank_call(&self) {
        self.k_distance_rank_calls.incr();
    }

    fn record_k_distance_rank_found(&self) {
        self.k_distance_rank_found.incr();
    }

    fn record_k_distance_rank_scan_step(&self) {
        self.k_distance_rank_scan_steps.incr();
    }
}
