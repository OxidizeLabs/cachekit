#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::ClockRing;

// Fuzz eviction patterns with reference bits
//
// Tests the second-chance algorithm by varying which entries are touched
// (referenced) before eviction, ensuring the CLOCK algorithm behaves correctly.
fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let capacity = (data[0] as usize % 20).max(2);
    let mut ring = ClockRing::new(capacity);

    // Fill the ring
    for i in 0..capacity {
        ring.insert(i as u32, i as u32);
    }

    let mut idx = 1;
    while idx < data.len() {
        if idx + 1 >= data.len() {
            break;
        }

        let key = data[idx] as u32 % capacity as u32;
        let should_touch = data[idx + 1] % 2 == 0;

        if should_touch {
            ring.touch(&key);
        }

        // Insert new entry to trigger eviction
        let new_key = capacity as u32 + (idx as u32);
        ring.insert(new_key, new_key);

        assert!(ring.len() <= ring.capacity());

        idx += 2;
    }
});
