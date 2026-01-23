#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::FrequencyBuckets;

// Fuzz arbitrary operation sequences on FrequencyBuckets
//
// Tests random sequences of insert, touch, remove, pop_min, peek_min,
// and clear operations to find edge cases and invariant violations.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

    let mut idx = 0;
    while idx < data.len() {
        if idx + 2 >= data.len() {
            break;
        }

        let op = data[idx] % 10;
        let key = u32::from(data[idx + 1]);
        let _val = u32::from(data[idx + 2]);

        match op {
            0 => {
                // Insert
                buckets.insert(key);
            }
            1 => {
                // Touch
                buckets.touch(&key);
            }
            2 => {
                // Remove
                buckets.remove(&key);
            }
            3 => {
                // Pop min
                buckets.pop_min();
            }
            4 => {
                // Peek min (read-only)
                let _ = buckets.peek_min();
            }
            5 => {
                // Check contains
                let _ = buckets.contains(&key);
            }
            6 => {
                // Check frequency
                let _ = buckets.frequency(&key);
            }
            7 => {
                // Check min_freq
                let _ = buckets.min_freq();
            }
            8 => {
                // Touch capped
                let max_freq = u64::from(data[idx + 2]).max(1);
                buckets.touch_capped(&key, max_freq);
            }
            9 => {
                // Clear
                buckets.clear();
            }
            _ => unreachable!(),
        }

        // Validate invariants after each operation
        buckets.debug_validate_invariants();

        // Check basic consistency
        if buckets.is_empty() {
            assert_eq!(buckets.len(), 0);
            assert_eq!(buckets.min_freq(), None);
            assert_eq!(buckets.peek_min(), None);
        } else {
            assert!(buckets.len() > 0);
            assert!(buckets.min_freq().is_some());
            assert!(buckets.peek_min().is_some());
        }

        // Frequency should be at least 1 for any present key
        if buckets.contains(&key) {
            assert!(buckets.frequency(&key).unwrap() >= 1);
        }

        idx += 3;
    }
});
