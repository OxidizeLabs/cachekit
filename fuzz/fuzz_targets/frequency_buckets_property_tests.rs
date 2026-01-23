#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::FrequencyBuckets;

// Fuzz property-based tests for FrequencyBuckets
//
// Tests specific invariants and properties:
// - Frequency monotonicity (touch always increments)
// - FIFO ordering within same frequency
// - min_freq accuracy
// - Pop/peek consistency
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let test_type = data[0] % 5;

    match test_type {
        0 => test_frequency_monotonicity(&data[1..]),
        1 => test_fifo_within_frequency(&data[1..]),
        2 => test_min_freq_accuracy(&data[1..]),
        3 => test_peek_pop_consistency(&data[1..]),
        4 => test_clear_operations(&data[1..]),
        _ => unreachable!(),
    }
});

// Property: touch() always increments frequency by 1 (or caps at u64::MAX)
fn test_frequency_monotonicity(data: &[u8]) {
    let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

    for &byte in data {
        let key = u32::from(byte);

        if !buckets.contains(&key) {
            buckets.insert(key);
            assert_eq!(buckets.frequency(&key), Some(1));
        }

        let old_freq = buckets.frequency(&key).unwrap();
        if old_freq < u64::MAX {
            let new_freq = buckets.touch(&key).unwrap();
            assert_eq!(new_freq, old_freq + 1);
            assert_eq!(buckets.frequency(&key), Some(new_freq));
        }

        buckets.debug_validate_invariants();
    }
}

// Property: FIFO ordering within same frequency bucket
fn test_fifo_within_frequency(data: &[u8]) {
    let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

    // Insert keys in order
    let keys: Vec<u32> = data.iter().map(|&b| u32::from(b)).collect();
    for &key in &keys {
        buckets.insert(key);
    }

    // All keys should be at freq=1
    // Pop should return them in insertion order (FIFO)
    let mut expected_order = keys.clone();
    expected_order.dedup();

    for expected_key in expected_order {
        if let Some((popped_key, freq)) = buckets.pop_min() {
            assert_eq!(popped_key, expected_key);
            assert_eq!(freq, 1);
        }
    }

    assert!(buckets.is_empty());
    buckets.debug_validate_invariants();
}

// Property: min_freq is always the minimum frequency present
fn test_min_freq_accuracy(data: &[u8]) {
    let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }

        let op = chunk[0] % 2;
        let key = u32::from(chunk[1]);

        match op {
            0 => {
                buckets.insert(key);
            }
            1 => {
                buckets.touch(&key);
            }
            _ => unreachable!(),
        }

        // Compute actual minimum frequency
        let mut actual_min: Option<u64> = None;
        for (_, meta) in buckets.iter_entries() {
            actual_min = match actual_min {
                None => Some(meta.freq),
                Some(min) => Some(min.min(meta.freq)),
            };
        }

        // Verify min_freq matches
        assert_eq!(buckets.min_freq(), actual_min);
        buckets.debug_validate_invariants();
    }
}

// Property: peek_min and pop_min return the same key/freq
fn test_peek_pop_consistency(data: &[u8]) {
    let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

    // Insert some keys
    for &byte in data.iter().take(20) {
        let key = u32::from(byte);
        buckets.insert(key);
    }

    // Touch some keys to vary frequencies
    for &byte in data.iter().skip(20).take(20) {
        let key = u32::from(byte);
        buckets.touch(&key);
    }

    // Peek and pop should be consistent
    while !buckets.is_empty() {
        let peeked = buckets.peek_min();
        let popped = buckets.pop_min();

        if let (Some((peek_key, peek_freq)), Some((pop_key, pop_freq))) = (peeked, popped) {
            assert_eq!(peek_key, pop_key);
            assert_eq!(peek_freq, pop_freq);
        }

        buckets.debug_validate_invariants();
    }

    assert_eq!(buckets.peek_min(), None);
    assert_eq!(buckets.pop_min(), None);
}

// Property: clear() resets all state correctly
fn test_clear_operations(data: &[u8]) {
    let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

    let mut idx = 0;
    while idx < data.len() {
        // Insert and touch some keys
        let batch_size = (data[idx] as usize % 10).min(data.len() - idx - 1);
        for i in 0..batch_size {
            if idx + i + 1 >= data.len() {
                break;
            }
            let key = u32::from(data[idx + i + 1]);
            buckets.insert(key);
            if i % 2 == 0 {
                buckets.touch(&key);
            }
        }

        idx += batch_size + 1;

        if idx >= data.len() {
            break;
        }

        // Choose clear operation
        let clear_op = data[idx] % 2;
        match clear_op {
            0 => buckets.clear(),
            1 => buckets.clear_shrink(),
            _ => unreachable!(),
        }

        // Verify empty state
        assert!(buckets.is_empty());
        assert_eq!(buckets.len(), 0);
        assert_eq!(buckets.min_freq(), None);
        assert_eq!(buckets.peek_min(), None);
        assert_eq!(buckets.pop_min(), None);

        buckets.debug_validate_invariants();

        idx += 1;
    }
}
