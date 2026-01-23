#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::ClockRing;

/// Fuzz stress test with many inserts
///
/// Tests behavior under heavy insert load with varying keys and values
/// to find capacity-related edge cases and eviction bugs.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let capacity = (data[0] as usize % 100).max(1);
    let mut ring = ClockRing::new(capacity);

    for chunk in data[1..].chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let key = chunk[0] as u32;
        let value = chunk[1] as u32;
        ring.insert(key, value);

        assert!(ring.len() <= ring.capacity());
    }

    ring.debug_validate_invariants();
});
