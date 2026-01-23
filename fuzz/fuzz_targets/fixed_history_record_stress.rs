#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::FixedHistory;

// Fuzz stress test with many record operations
//
// Tests behavior under heavy recording load with varying timestamps
// to find wrapping edge cases and ensure order is maintained.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Use first byte to determine capacity (1-16)
    let capacity_byte = data[0] % 16;
    let capacity = (capacity_byte as usize).max(1);

    // Route to different capacity implementations
    match capacity {
        1 => stress_test::<1>(&data[1..]),
        2 => stress_test::<2>(&data[1..]),
        3 => stress_test::<3>(&data[1..]),
        4 => stress_test::<4>(&data[1..]),
        5 => stress_test::<5>(&data[1..]),
        6 => stress_test::<6>(&data[1..]),
        7 => stress_test::<7>(&data[1..]),
        8 => stress_test::<8>(&data[1..]),
        9 => stress_test::<9>(&data[1..]),
        10 => stress_test::<10>(&data[1..]),
        11 => stress_test::<11>(&data[1..]),
        12 => stress_test::<12>(&data[1..]),
        13 => stress_test::<13>(&data[1..]),
        14 => stress_test::<14>(&data[1..]),
        15 => stress_test::<15>(&data[1..]),
        _ => stress_test::<16>(&data[1..]),
    }
});

fn stress_test<const K: usize>(data: &[u8]) {
    let mut history = FixedHistory::<K>::new();
    let mut last_k_values: Vec<u64> = Vec::new();

    for &byte in data {
        let timestamp = u64::from(byte);

        // Record and track in reference implementation
        history.record(timestamp);
        last_k_values.push(timestamp);

        // Keep only last K values in reference
        if last_k_values.len() > K {
            last_k_values.remove(0);
        }

        // Validate invariants
        history.debug_validate_invariants();
        assert!(history.len() <= K);
        assert_eq!(history.len(), last_k_values.len());

        // Verify most recent
        if let Some(most_recent) = history.most_recent() {
            assert_eq!(most_recent, *last_k_values.last().unwrap());
        }

        // Verify to_vec_mru matches reference (in reverse)
        let vec_mru = history.to_vec_mru();
        let expected: Vec<u64> = last_k_values.iter().rev().copied().collect();
        assert_eq!(vec_mru, expected);

        // Verify each kth_most_recent
        for k in 1..=history.len() {
            let result = history.kth_most_recent(k);
            let expected_idx = last_k_values.len() - k;
            assert_eq!(result, Some(last_k_values[expected_idx]));
        }
    }

    // Final comprehensive check
    history.debug_validate_invariants();
    assert_eq!(history.len(), last_k_values.len().min(K));
}
