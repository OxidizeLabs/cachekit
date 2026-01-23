#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::FixedHistory;

// Fuzz property-based tests for FixedHistory
//
// Tests specific invariants and properties:
// - Order preservation after wrapping
// - kth_most_recent consistency
// - Boundary conditions (k=0, k>len)
// - Empty state consistency
fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let capacity_byte = data[0] % 16;
    let capacity = (capacity_byte as usize).max(1);
    let test_type = data[1] % 4;

    match capacity {
        1 => run_property_test::<1>(&data[2..], test_type),
        2 => run_property_test::<2>(&data[2..], test_type),
        3 => run_property_test::<3>(&data[2..], test_type),
        4 => run_property_test::<4>(&data[2..], test_type),
        5 => run_property_test::<5>(&data[2..], test_type),
        6 => run_property_test::<6>(&data[2..], test_type),
        7 => run_property_test::<7>(&data[2..], test_type),
        8 => run_property_test::<8>(&data[2..], test_type),
        9 => run_property_test::<9>(&data[2..], test_type),
        10 => run_property_test::<10>(&data[2..], test_type),
        11 => run_property_test::<11>(&data[2..], test_type),
        12 => run_property_test::<12>(&data[2..], test_type),
        13 => run_property_test::<13>(&data[2..], test_type),
        14 => run_property_test::<14>(&data[2..], test_type),
        15 => run_property_test::<15>(&data[2..], test_type),
        _ => run_property_test::<16>(&data[2..], test_type),
    }
});

fn run_property_test<const K: usize>(data: &[u8], test_type: u8) {
    match test_type {
        0 => test_order_preservation::<K>(data),
        1 => test_boundary_conditions::<K>(data),
        2 => test_wrap_consistency::<K>(data),
        3 => test_clear_operations::<K>(data),
        _ => unreachable!(),
    }
}

// Property: Order is preserved - most recent is always the last recorded
fn test_order_preservation<const K: usize>(data: &[u8]) {
    let mut history = FixedHistory::<K>::new();

    for &byte in data {
        let timestamp = u64::from(byte);
        history.record(timestamp);

        // Most recent should always be the last recorded
        assert_eq!(history.most_recent(), Some(timestamp));

        // Verify MRU order is maintained
        let vec = history.to_vec_mru();
        if !vec.is_empty() {
            assert_eq!(vec[0], timestamp);

            // Each subsequent element should be accessible via kth_most_recent
            for (i, &val) in vec.iter().enumerate() {
                assert_eq!(history.kth_most_recent(i + 1), Some(val));
            }
        }

        history.debug_validate_invariants();
    }
}

// Property: Boundary conditions behave correctly
fn test_boundary_conditions<const K: usize>(data: &[u8]) {
    let mut history = FixedHistory::<K>::new();

    // Test with empty history
    assert_eq!(history.kth_most_recent(0), None);
    assert_eq!(history.kth_most_recent(1), None);
    assert_eq!(history.kth_most_recent(K), None);
    assert_eq!(history.kth_most_recent(K + 1), None);

    // Record some values
    for &byte in data.iter().take(K.min(data.len())) {
        history.record(u64::from(byte));
    }

    let len = history.len();

    // k=0 always returns None
    assert_eq!(history.kth_most_recent(0), None);

    // k > len returns None
    assert_eq!(history.kth_most_recent(len + 1), None);
    assert_eq!(history.kth_most_recent(K + 10), None);

    // Valid k values (1..=len) should return Some
    for k in 1..=len {
        assert!(history.kth_most_recent(k).is_some());
    }

    history.debug_validate_invariants();
}

// Property: Wrapping maintains consistent state
fn test_wrap_consistency<const K: usize>(data: &[u8]) {
    if K == 0 {
        return;
    }

    let mut history = FixedHistory::<K>::new();

    // Record more than capacity to force wrapping
    for &byte in data {
        let timestamp = u64::from(byte);
        let old_len = history.len();

        history.record(timestamp);

        // Length increases until capacity is reached
        if old_len < K {
            assert_eq!(history.len(), old_len + 1);
        } else {
            assert_eq!(history.len(), K);
        }

        // After wrapping, len should stay at K
        assert!(history.len() <= K);

        // Most recent should always be the last recorded
        assert_eq!(history.most_recent(), Some(timestamp));

        history.debug_validate_invariants();
    }

    // After many wraps, verify consistency
    if history.len() > 0 {
        let vec = history.to_vec_mru();
        assert_eq!(vec.len(), history.len());

        // Verify order consistency
        for k in 1..=history.len() {
            assert_eq!(history.kth_most_recent(k), Some(vec[k - 1]));
        }
    }
}

// Property: Clear operations reset state correctly
fn test_clear_operations<const K: usize>(data: &[u8]) {
    let mut history = FixedHistory::<K>::new();

    let mut idx = 0;
    while idx < data.len() {
        // Record some values
        let record_count = (data[idx] as usize % 10).min(data.len() - idx - 1);
        for i in 0..record_count {
            if idx + i + 1 >= data.len() {
                break;
            }
            history.record(u64::from(data[idx + i + 1]));
        }

        idx += record_count + 1;

        if idx >= data.len() {
            break;
        }

        // Choose clear operation
        let clear_op = data[idx] % 2;
        match clear_op {
            0 => history.clear(),
            1 => history.clear_shrink(),
            _ => unreachable!(),
        }

        // Verify empty state
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.most_recent(), None);
        assert_eq!(history.to_vec_mru(), Vec::<u64>::new());
        assert_eq!(history.kth_most_recent(1), None);

        history.debug_validate_invariants();

        idx += 1;
    }
}
