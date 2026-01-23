#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::FixedHistory;

// Fuzz arbitrary operation sequences on FixedHistory
//
// Tests random sequences of record, kth_most_recent, most_recent, clear,
// and to_vec_mru operations to find edge cases and invariant violations.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Use first byte to determine capacity (1-32)
    let capacity_byte = data[0] % 32;
    let capacity = (capacity_byte as usize).max(1);

    // Route to different capacity implementations
    match capacity {
        1 => fuzz_with_capacity::<1>(&data[1..]),
        2 => fuzz_with_capacity::<2>(&data[1..]),
        3 => fuzz_with_capacity::<3>(&data[1..]),
        4 => fuzz_with_capacity::<4>(&data[1..]),
        5 => fuzz_with_capacity::<5>(&data[1..]),
        6 => fuzz_with_capacity::<6>(&data[1..]),
        7 => fuzz_with_capacity::<7>(&data[1..]),
        8 => fuzz_with_capacity::<8>(&data[1..]),
        9 => fuzz_with_capacity::<9>(&data[1..]),
        10 => fuzz_with_capacity::<10>(&data[1..]),
        11 => fuzz_with_capacity::<11>(&data[1..]),
        12 => fuzz_with_capacity::<12>(&data[1..]),
        13 => fuzz_with_capacity::<13>(&data[1..]),
        14 => fuzz_with_capacity::<14>(&data[1..]),
        15 => fuzz_with_capacity::<15>(&data[1..]),
        16 => fuzz_with_capacity::<16>(&data[1..]),
        _ => fuzz_with_capacity::<32>(&data[1..]),
    }
});

fn fuzz_with_capacity<const K: usize>(data: &[u8]) {
    let mut history = FixedHistory::<K>::new();

    let mut idx = 0;
    while idx < data.len() {
        if idx + 1 >= data.len() {
            break;
        }

        let op = data[idx] % 6;
        let timestamp = u64::from(data[idx + 1]);

        match op {
            0 => {
                // Record timestamp
                history.record(timestamp);
            }
            1 => {
                // Get most recent
                let result = history.most_recent();
                // Verify consistency
                if !history.is_empty() {
                    assert!(result.is_some());
                }
            }
            2 => {
                // Get kth most recent
                let k = ((data[idx] as usize) % K.max(1)).max(1);
                let result = history.kth_most_recent(k);
                // k out of bounds should return None
                if k > history.len() {
                    assert!(result.is_none());
                }
            }
            3 => {
                // to_vec_mru
                let vec = history.to_vec_mru();
                assert_eq!(vec.len(), history.len());

                // Verify order - each element should match kth_most_recent
                for (i, &val) in vec.iter().enumerate() {
                    assert_eq!(history.kth_most_recent(i + 1), Some(val));
                }
            }
            4 => {
                // Clear
                history.clear();
                assert!(history.is_empty());
                assert_eq!(history.len(), 0);
                assert_eq!(history.most_recent(), None);
            }
            5 => {
                // Clear and shrink (equivalent to clear for FixedHistory)
                history.clear_shrink();
                assert!(history.is_empty());
            }
            _ => unreachable!(),
        }

        // Validate invariants after each operation
        history.debug_validate_invariants();
        assert!(history.len() <= K);
        assert_eq!(history.capacity(), K);

        // Verify empty state consistency
        if history.is_empty() {
            assert_eq!(history.len(), 0);
            assert_eq!(history.most_recent(), None);
            assert_eq!(history.to_vec_mru().len(), 0);
        }

        idx += 2;
    }
}
