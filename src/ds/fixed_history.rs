#[derive(Debug, Clone)]
pub struct FixedHistory<const K: usize> {
    data: [u64; K],
    len: usize,
    cursor: usize,
}

impl<const K: usize> FixedHistory<K> {
    pub fn new() -> Self {
        Self {
            data: [0; K],
            len: 0,
            cursor: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        K
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn record(&mut self, timestamp: u64) {
        if K == 0 {
            return;
        }
        self.data[self.cursor] = timestamp;
        self.cursor = (self.cursor + 1) % K;
        if self.len < K {
            self.len += 1;
        }
    }

    pub fn most_recent(&self) -> Option<u64> {
        self.kth_most_recent(1)
    }

    pub fn kth_most_recent(&self, k: usize) -> Option<u64> {
        if K == 0 || k == 0 || k > self.len {
            return None;
        }
        let idx = (self.cursor + K - k) % K;
        Some(self.data[idx])
    }

    pub fn to_vec_mru(&self) -> Vec<u64> {
        (1..=self.len)
            .filter_map(|k| self.kth_most_recent(k))
            .collect()
    }
}

impl<const K: usize> Default for FixedHistory<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_history_tracks_last_k() {
        let mut history = FixedHistory::<3>::new();
        history.record(10);
        history.record(20);
        history.record(30);
        assert_eq!(history.to_vec_mru(), vec![30, 20, 10]);

        history.record(40);
        assert_eq!(history.to_vec_mru(), vec![40, 30, 20]);
        assert_eq!(history.kth_most_recent(3), Some(20));
    }
}
