use crate::metrics::cell::MetricsCell;
use crate::metrics::traits::{
    ArcMetricsRecorder, CarMetricsRecorder, ClockMetricsRecorder, ClockProMetricsRecorder,
    CoreMetricsRecorder, FifoMetricsReadRecorder, FifoMetricsRecorder, LfuMetricsReadRecorder,
    LfuMetricsRecorder, LruKMetricsReadRecorder, LruKMetricsRecorder, LruMetricsReadRecorder,
    LruMetricsRecorder, MfuMetricsReadRecorder, MfuMetricsRecorder, NruMetricsRecorder,
    S3FifoMetricsRecorder, SlruMetricsRecorder, TwoQMetricsRecorder,
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

// ---------------------------------------------------------------------------
// CoreOnlyMetrics (LIFO, Random)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct CoreOnlyMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
}

impl CoreMetricsRecorder for CoreOnlyMetrics {
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

// ---------------------------------------------------------------------------
// ArcMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct ArcMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub t1_to_t2_promotions: u64,
    pub b1_ghost_hits: u64,
    pub b2_ghost_hits: u64,
    pub p_increases: u64,
    pub p_decreases: u64,
    pub t1_evictions: u64,
    pub t2_evictions: u64,
}

impl CoreMetricsRecorder for ArcMetrics {
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

impl ArcMetricsRecorder for ArcMetrics {
    fn record_t1_to_t2_promotion(&mut self) {
        self.t1_to_t2_promotions += 1;
    }
    fn record_b1_ghost_hit(&mut self) {
        self.b1_ghost_hits += 1;
    }
    fn record_b2_ghost_hit(&mut self) {
        self.b2_ghost_hits += 1;
    }
    fn record_p_increase(&mut self) {
        self.p_increases += 1;
    }
    fn record_p_decrease(&mut self) {
        self.p_decreases += 1;
    }
    fn record_t1_eviction(&mut self) {
        self.t1_evictions += 1;
    }
    fn record_t2_eviction(&mut self) {
        self.t2_evictions += 1;
    }
}

// ---------------------------------------------------------------------------
// CarMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct CarMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub recent_to_frequent_promotions: u64,
    pub ghost_recent_hits: u64,
    pub ghost_frequent_hits: u64,
    pub target_increases: u64,
    pub target_decreases: u64,
    pub hand_sweeps: u64,
}

impl CoreMetricsRecorder for CarMetrics {
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

impl CarMetricsRecorder for CarMetrics {
    fn record_recent_to_frequent_promotion(&mut self) {
        self.recent_to_frequent_promotions += 1;
    }
    fn record_ghost_recent_hit(&mut self) {
        self.ghost_recent_hits += 1;
    }
    fn record_ghost_frequent_hit(&mut self) {
        self.ghost_frequent_hits += 1;
    }
    fn record_target_increase(&mut self) {
        self.target_increases += 1;
    }
    fn record_target_decrease(&mut self) {
        self.target_decreases += 1;
    }
    fn record_hand_sweep(&mut self) {
        self.hand_sweeps += 1;
    }
}

// ---------------------------------------------------------------------------
// ClockMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct ClockMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub hand_advances: u64,
    pub ref_bit_resets: u64,
}

impl CoreMetricsRecorder for ClockMetrics {
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

impl ClockMetricsRecorder for ClockMetrics {
    fn record_hand_advance(&mut self) {
        self.hand_advances += 1;
    }
    fn record_ref_bit_reset(&mut self) {
        self.ref_bit_resets += 1;
    }
}

// ---------------------------------------------------------------------------
// ClockProMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct ClockProMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub cold_to_hot_promotions: u64,
    pub hot_to_cold_demotions: u64,
    pub test_insertions: u64,
    pub test_hits: u64,
}

impl CoreMetricsRecorder for ClockProMetrics {
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

impl ClockProMetricsRecorder for ClockProMetrics {
    fn record_cold_to_hot_promotion(&mut self) {
        self.cold_to_hot_promotions += 1;
    }
    fn record_hot_to_cold_demotion(&mut self) {
        self.hot_to_cold_demotions += 1;
    }
    fn record_test_insertion(&mut self) {
        self.test_insertions += 1;
    }
    fn record_test_hit(&mut self) {
        self.test_hits += 1;
    }
}

// ---------------------------------------------------------------------------
// MfuMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct MfuMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub pop_mfu_calls: u64,
    pub pop_mfu_found: u64,
    pub peek_mfu_calls: MetricsCell,
    pub peek_mfu_found: MetricsCell,
    pub frequency_calls: MetricsCell,
    pub frequency_found: MetricsCell,
}

impl CoreMetricsRecorder for MfuMetrics {
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

impl MfuMetricsRecorder for MfuMetrics {
    fn record_pop_mfu_call(&mut self) {
        self.pop_mfu_calls += 1;
    }
    fn record_pop_mfu_found(&mut self) {
        self.pop_mfu_found += 1;
    }
    fn record_peek_mfu_call(&mut self) {
        self.peek_mfu_calls.incr();
    }
    fn record_peek_mfu_found(&mut self) {
        self.peek_mfu_found.incr();
    }
    fn record_frequency_call(&mut self) {
        self.frequency_calls.incr();
    }
    fn record_frequency_found(&mut self) {
        self.frequency_found.incr();
    }
}

impl MfuMetricsReadRecorder for &MfuMetrics {
    fn record_peek_mfu_call(&self) {
        self.peek_mfu_calls.incr();
    }
    fn record_peek_mfu_found(&self) {
        self.peek_mfu_found.incr();
    }
    fn record_frequency_call(&self) {
        self.frequency_calls.incr();
    }
    fn record_frequency_found(&self) {
        self.frequency_found.incr();
    }
}

// ---------------------------------------------------------------------------
// NruMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct NruMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub sweep_steps: u64,
    pub ref_bit_resets: u64,
}

impl CoreMetricsRecorder for NruMetrics {
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

impl NruMetricsRecorder for NruMetrics {
    fn record_sweep_step(&mut self) {
        self.sweep_steps += 1;
    }
    fn record_ref_bit_reset(&mut self) {
        self.ref_bit_resets += 1;
    }
}

// ---------------------------------------------------------------------------
// SlruMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct SlruMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub probationary_to_protected: u64,
    pub protected_evictions: u64,
}

impl CoreMetricsRecorder for SlruMetrics {
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

impl SlruMetricsRecorder for SlruMetrics {
    fn record_probationary_to_protected(&mut self) {
        self.probationary_to_protected += 1;
    }
    fn record_protected_eviction(&mut self) {
        self.protected_evictions += 1;
    }
}

// ---------------------------------------------------------------------------
// TwoQMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct TwoQMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub a1in_to_am_promotions: u64,
    pub a1out_ghost_hits: u64,
}

impl CoreMetricsRecorder for TwoQMetrics {
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

impl TwoQMetricsRecorder for TwoQMetrics {
    fn record_a1in_to_am_promotion(&mut self) {
        self.a1in_to_am_promotions += 1;
    }
    fn record_a1out_ghost_hit(&mut self) {
        self.a1out_ghost_hits += 1;
    }
}

// ---------------------------------------------------------------------------
// S3FifoMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct S3FifoMetrics {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
    pub promotions: u64,
    pub main_reinserts: u64,
    pub small_evictions: u64,
    pub main_evictions: u64,
    pub ghost_hits: u64,
}

impl CoreMetricsRecorder for S3FifoMetrics {
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

impl S3FifoMetricsRecorder for S3FifoMetrics {
    fn record_promotion(&mut self) {
        self.promotions += 1;
    }
    fn record_main_reinsert(&mut self) {
        self.main_reinserts += 1;
    }
    fn record_small_eviction(&mut self) {
        self.small_evictions += 1;
    }
    fn record_main_eviction(&mut self) {
        self.main_evictions += 1;
    }
    fn record_ghost_hit(&mut self) {
        self.ghost_hits += 1;
    }
}
