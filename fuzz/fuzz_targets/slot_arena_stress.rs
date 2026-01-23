#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::SlotArena;
use std::collections::HashMap;

// Fuzz stress test with heavy operations and reference validation
//
// Tests behavior under heavy load with reference HashMap implementation
// to ensure correct slot reuse and value tracking.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut arena: SlotArena<u32> = SlotArena::new();
    let mut reference: HashMap<usize, u32> = HashMap::new();

    for (idx, &byte) in data.iter().enumerate() {
        let value = u32::from(byte);
        let op = idx % 3;

        match op {
            0 => {
                // insert
                let id = arena.insert(value);
                reference.insert(id.index(), value);
            }
            1 => {
                // remove a random existing slot
                if !reference.is_empty() {
                    let keys: Vec<_> = reference.keys().copied().collect();
                    let key_idx = (value as usize) % keys.len();
                    let key = keys[key_idx];

                    let id = cachekit::ds::SlotId(key);
                    let arena_val = arena.remove(id);
                    let ref_val = reference.remove(&key);

                    assert_eq!(arena_val, ref_val);
                }
            }
            2 => {
                // verify consistency
                for (&key, &expected_value) in &reference {
                    let id = cachekit::ds::SlotId(key);
                    assert_eq!(arena.get(id), Some(&expected_value));
                    assert!(arena.contains(id));
                }
            }
            _ => unreachable!(),
        }

        // Verify length matches
        assert_eq!(arena.len(), reference.len());
        assert_eq!(arena.is_empty(), reference.is_empty());
    }
});
