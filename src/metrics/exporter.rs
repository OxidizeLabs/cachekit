use crate::metrics::snapshot::{
    CacheMetricsSnapshot, LfuMetricsSnapshot, LruKMetricsSnapshot, LruMetricsSnapshot,
};
use crate::metrics::traits::MetricsExporter;
use std::io::Write;
use std::sync::Mutex;

/// Prometheus text exporter for cache metrics snapshots.
///
/// This exporter writes in the Prometheus text exposition format so it can be
/// scraped by Prometheus or forwarded to an OpenTelemetry collector.
#[derive(Debug)]
pub struct PrometheusTextExporter<W: Write + Send + Sync> {
    prefix: String,
    writer: Mutex<W>,
}

impl<W: Write + Send + Sync> PrometheusTextExporter<W> {
    pub fn new(prefix: impl Into<String>, writer: W) -> Self {
        Self {
            prefix: prefix.into(),
            writer: Mutex::new(writer),
        }
    }

    fn write_counter(&self, name: &str, value: u64) {
        let mut writer = self
            .writer
            .lock()
            .expect("metrics exporter writer poisoned");
        let _ = writeln!(writer, "# TYPE {} counter", name);
        let _ = writeln!(writer, "{} {}", name, value);
    }

    fn write_gauge(&self, name: &str, value: u64) {
        let mut writer = self
            .writer
            .lock()
            .expect("metrics exporter writer poisoned");
        let _ = writeln!(writer, "# TYPE {} gauge", name);
        let _ = writeln!(writer, "{} {}", name, value);
    }

    fn metric_name(&self, suffix: &str) -> String {
        if self.prefix.is_empty() {
            suffix.to_string()
        } else {
            format!("{}_{}", self.prefix, suffix)
        }
    }
}

impl<W: Write + Send + Sync> MetricsExporter<CacheMetricsSnapshot> for PrometheusTextExporter<W> {
    fn export(&self, snapshot: &CacheMetricsSnapshot) {
        self.write_counter(&self.metric_name("get_calls_total"), snapshot.get_calls);
        self.write_counter(&self.metric_name("get_hits_total"), snapshot.get_hits);
        self.write_counter(&self.metric_name("get_misses_total"), snapshot.get_misses);
        self.write_counter(
            &self.metric_name("insert_calls_total"),
            snapshot.insert_calls,
        );
        self.write_counter(
            &self.metric_name("insert_updates_total"),
            snapshot.insert_updates,
        );
        self.write_counter(&self.metric_name("insert_new_total"), snapshot.insert_new);
        self.write_counter(&self.metric_name("evict_calls_total"), snapshot.evict_calls);
        self.write_counter(
            &self.metric_name("evicted_entries_total"),
            snapshot.evicted_entries,
        );
        self.write_counter(&self.metric_name("stale_skips_total"), snapshot.stale_skips);
        self.write_counter(
            &self.metric_name("evict_scan_steps_total"),
            snapshot.evict_scan_steps,
        );
        self.write_counter(
            &self.metric_name("pop_oldest_calls_total"),
            snapshot.pop_oldest_calls,
        );
        self.write_counter(
            &self.metric_name("pop_oldest_found_total"),
            snapshot.pop_oldest_found,
        );
        self.write_counter(
            &self.metric_name("pop_oldest_empty_or_stale_total"),
            snapshot.pop_oldest_empty_or_stale,
        );
        self.write_counter(
            &self.metric_name("peek_oldest_calls_total"),
            snapshot.peek_oldest_calls,
        );
        self.write_counter(
            &self.metric_name("peek_oldest_found_total"),
            snapshot.peek_oldest_found,
        );
        self.write_counter(
            &self.metric_name("age_rank_calls_total"),
            snapshot.age_rank_calls,
        );
        self.write_counter(
            &self.metric_name("age_rank_found_total"),
            snapshot.age_rank_found,
        );
        self.write_counter(
            &self.metric_name("age_rank_scan_steps_total"),
            snapshot.age_rank_scan_steps,
        );
        self.write_gauge(&self.metric_name("cache_len"), snapshot.cache_len as u64);
        self.write_gauge(
            &self.metric_name("insertion_order_len"),
            snapshot.insertion_order_len as u64,
        );
        self.write_gauge(&self.metric_name("capacity"), snapshot.capacity as u64);
    }
}

impl<W: Write + Send + Sync> MetricsExporter<LruMetricsSnapshot> for PrometheusTextExporter<W> {
    fn export(&self, snapshot: &LruMetricsSnapshot) {
        self.write_counter(&self.metric_name("get_calls_total"), snapshot.get_calls);
        self.write_counter(&self.metric_name("get_hits_total"), snapshot.get_hits);
        self.write_counter(&self.metric_name("get_misses_total"), snapshot.get_misses);
        self.write_counter(
            &self.metric_name("insert_calls_total"),
            snapshot.insert_calls,
        );
        self.write_counter(
            &self.metric_name("insert_updates_total"),
            snapshot.insert_updates,
        );
        self.write_counter(&self.metric_name("insert_new_total"), snapshot.insert_new);
        self.write_counter(&self.metric_name("evict_calls_total"), snapshot.evict_calls);
        self.write_counter(
            &self.metric_name("evicted_entries_total"),
            snapshot.evicted_entries,
        );
        self.write_counter(
            &self.metric_name("pop_lru_calls_total"),
            snapshot.pop_lru_calls,
        );
        self.write_counter(
            &self.metric_name("pop_lru_found_total"),
            snapshot.pop_lru_found,
        );
        self.write_counter(
            &self.metric_name("peek_lru_calls_total"),
            snapshot.peek_lru_calls,
        );
        self.write_counter(
            &self.metric_name("peek_lru_found_total"),
            snapshot.peek_lru_found,
        );
        self.write_counter(&self.metric_name("touch_calls_total"), snapshot.touch_calls);
        self.write_counter(&self.metric_name("touch_found_total"), snapshot.touch_found);
        self.write_counter(
            &self.metric_name("recency_rank_calls_total"),
            snapshot.recency_rank_calls,
        );
        self.write_counter(
            &self.metric_name("recency_rank_found_total"),
            snapshot.recency_rank_found,
        );
        self.write_counter(
            &self.metric_name("recency_rank_scan_steps_total"),
            snapshot.recency_rank_scan_steps,
        );
        self.write_gauge(&self.metric_name("cache_len"), snapshot.cache_len as u64);
        self.write_gauge(&self.metric_name("capacity"), snapshot.capacity as u64);
    }
}

impl<W: Write + Send + Sync> MetricsExporter<LfuMetricsSnapshot> for PrometheusTextExporter<W> {
    fn export(&self, snapshot: &LfuMetricsSnapshot) {
        self.write_counter(&self.metric_name("get_calls_total"), snapshot.get_calls);
        self.write_counter(&self.metric_name("get_hits_total"), snapshot.get_hits);
        self.write_counter(&self.metric_name("get_misses_total"), snapshot.get_misses);
        self.write_counter(
            &self.metric_name("insert_calls_total"),
            snapshot.insert_calls,
        );
        self.write_counter(
            &self.metric_name("insert_updates_total"),
            snapshot.insert_updates,
        );
        self.write_counter(&self.metric_name("insert_new_total"), snapshot.insert_new);
        self.write_counter(&self.metric_name("evict_calls_total"), snapshot.evict_calls);
        self.write_counter(
            &self.metric_name("evicted_entries_total"),
            snapshot.evicted_entries,
        );
        self.write_counter(
            &self.metric_name("pop_lfu_calls_total"),
            snapshot.pop_lfu_calls,
        );
        self.write_counter(
            &self.metric_name("pop_lfu_found_total"),
            snapshot.pop_lfu_found,
        );
        self.write_counter(
            &self.metric_name("peek_lfu_calls_total"),
            snapshot.peek_lfu_calls,
        );
        self.write_counter(
            &self.metric_name("peek_lfu_found_total"),
            snapshot.peek_lfu_found,
        );
        self.write_counter(
            &self.metric_name("frequency_calls_total"),
            snapshot.frequency_calls,
        );
        self.write_counter(
            &self.metric_name("frequency_found_total"),
            snapshot.frequency_found,
        );
        self.write_counter(
            &self.metric_name("reset_frequency_calls_total"),
            snapshot.reset_frequency_calls,
        );
        self.write_counter(
            &self.metric_name("reset_frequency_found_total"),
            snapshot.reset_frequency_found,
        );
        self.write_counter(
            &self.metric_name("increment_frequency_calls_total"),
            snapshot.increment_frequency_calls,
        );
        self.write_counter(
            &self.metric_name("increment_frequency_found_total"),
            snapshot.increment_frequency_found,
        );
        self.write_gauge(&self.metric_name("cache_len"), snapshot.cache_len as u64);
        self.write_gauge(&self.metric_name("capacity"), snapshot.capacity as u64);
    }
}

impl<W: Write + Send + Sync> MetricsExporter<LruKMetricsSnapshot> for PrometheusTextExporter<W> {
    fn export(&self, snapshot: &LruKMetricsSnapshot) {
        self.write_counter(&self.metric_name("get_calls_total"), snapshot.get_calls);
        self.write_counter(&self.metric_name("get_hits_total"), snapshot.get_hits);
        self.write_counter(&self.metric_name("get_misses_total"), snapshot.get_misses);
        self.write_counter(
            &self.metric_name("insert_calls_total"),
            snapshot.insert_calls,
        );
        self.write_counter(
            &self.metric_name("insert_updates_total"),
            snapshot.insert_updates,
        );
        self.write_counter(&self.metric_name("insert_new_total"), snapshot.insert_new);
        self.write_counter(&self.metric_name("evict_calls_total"), snapshot.evict_calls);
        self.write_counter(
            &self.metric_name("evicted_entries_total"),
            snapshot.evicted_entries,
        );
        self.write_counter(
            &self.metric_name("pop_lru_calls_total"),
            snapshot.pop_lru_calls,
        );
        self.write_counter(
            &self.metric_name("pop_lru_found_total"),
            snapshot.pop_lru_found,
        );
        self.write_counter(
            &self.metric_name("peek_lru_calls_total"),
            snapshot.peek_lru_calls,
        );
        self.write_counter(
            &self.metric_name("peek_lru_found_total"),
            snapshot.peek_lru_found,
        );
        self.write_counter(&self.metric_name("touch_calls_total"), snapshot.touch_calls);
        self.write_counter(&self.metric_name("touch_found_total"), snapshot.touch_found);
        self.write_counter(
            &self.metric_name("recency_rank_calls_total"),
            snapshot.recency_rank_calls,
        );
        self.write_counter(
            &self.metric_name("recency_rank_found_total"),
            snapshot.recency_rank_found,
        );
        self.write_counter(
            &self.metric_name("recency_rank_scan_steps_total"),
            snapshot.recency_rank_scan_steps,
        );
        self.write_counter(
            &self.metric_name("pop_lru_k_calls_total"),
            snapshot.pop_lru_k_calls,
        );
        self.write_counter(
            &self.metric_name("pop_lru_k_found_total"),
            snapshot.pop_lru_k_found,
        );
        self.write_counter(
            &self.metric_name("peek_lru_k_calls_total"),
            snapshot.peek_lru_k_calls,
        );
        self.write_counter(
            &self.metric_name("peek_lru_k_found_total"),
            snapshot.peek_lru_k_found,
        );
        self.write_counter(
            &self.metric_name("k_distance_calls_total"),
            snapshot.k_distance_calls,
        );
        self.write_counter(
            &self.metric_name("k_distance_found_total"),
            snapshot.k_distance_found,
        );
        self.write_counter(
            &self.metric_name("k_distance_rank_calls_total"),
            snapshot.k_distance_rank_calls,
        );
        self.write_counter(
            &self.metric_name("k_distance_rank_found_total"),
            snapshot.k_distance_rank_found,
        );
        self.write_counter(
            &self.metric_name("k_distance_rank_scan_steps_total"),
            snapshot.k_distance_rank_scan_steps,
        );
        self.write_gauge(&self.metric_name("cache_len"), snapshot.cache_len as u64);
        self.write_gauge(&self.metric_name("capacity"), snapshot.capacity as u64);
    }
}
