#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::FrequencyBuckets;
use std::collections::HashMap;

// Fuzz stress test with heavy insert and touch operations
//
// Tests behavior under heavy LFU tracking load with reference implementation
// validation to ensure frequency tracking, min_freq, and eviction order are correct.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
    let mut reference: HashMap<u32, u64> = HashMap::new();

    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }

        let op = chunk[0] % 3;
        let key = u32::from(chunk[1]);

        match op {
            0 => {
                // Insert
                if buckets.insert(key) {
                    reference.insert(key, 1);
                }
            }
            1 => {
                // Touch
                if let Some(new_freq) = buckets.touch(&key) {
                    *reference.get_mut(&key).unwrap() = new_freq;
                }
            }
            2 => {
                // Pop min
                if let Some((evicted_key, evicted_freq)) = buckets.pop_min() {
                    assert_eq!(reference.remove(&evicted_key), Some(evicted_freq));
                }
            }
            _ => unreachable!(),
        }

        // Validate invariants
        buckets.debug_validate_invariants();

        // Verify len matches
        assert_eq!(buckets.len(), reference.len());

        // Verify each key's frequency
        for (key, expected_freq) in &reference {
            assert_eq!(buckets.frequency(key), Some(*expected_freq));
            assert!(buckets.contains(key));
        }

        // Verify min_freq is correct
        if !reference.is_empty() {
            let min_in_reference = reference.values().min().copied().unwrap();
            assert_eq!(buckets.min_freq(), Some(min_in_reference));
        } else {
            assert_eq!(buckets.min_freq(), None);
        }
    }

    // Final validation
    buckets.debug_validate_invariants();
    assert_eq!(buckets.len(), reference.len());
});
