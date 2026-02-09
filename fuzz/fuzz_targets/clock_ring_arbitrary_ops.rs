#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::ClockRing;

// Fuzz arbitrary operation sequences on ClockRing
//
// Tests random sequences of insert, get, peek, touch, update, remove, and pop_victim
// operations to find edge cases and invariant violations.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let capacity = (data[0] as usize % 50).max(1);
    let mut ring = ClockRing::new(capacity);

    let mut idx = 1;
    while idx < data.len() {
        if idx + 2 >= data.len() {
            break;
        }

        let op = data[idx] % 7;
        let key = data[idx + 1] as u32;
        let value = data[idx + 2] as u32;

        match op {
            0 => {
                ring.insert(key, value);
            }
            1 => {
                let _ = ring.get(&key);
            }
            2 => {
                ring.peek(&key);
            }
            3 => {
                ring.touch(&key);
            }
            4 => {
                ring.update(&key, value);
            }
            5 => {
                ring.remove(&key);
            }
            6 => {
                ring.pop_victim();
            }
            _ => unreachable!(),
        }

        // Validate basic invariant
        assert!(ring.len() <= ring.capacity());

        idx += 3;
    }
});
