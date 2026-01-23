#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::{SlotArena, SlotId};
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
    let mut reference: HashMap<SlotId, u32> = HashMap::new();

    for (idx, &byte) in data.iter().enumerate() {
        let value = u32::from(byte);
        let op = idx % 3;

        match op {
            0 => {
                // insert
                let id = arena.insert(value);
                reference.insert(id, value);
            }
            1 => {
                // remove a random existing slot
                if !reference.is_empty() {
                    let keys: Vec<_> = reference.keys().copied().collect();
                    let key_idx = (value as usize) % keys.len();
                    let id = keys[key_idx];
                    let arena_val = arena.remove(id);
                    let ref_val = reference.remove(&id);

                    assert_eq!(arena_val, ref_val);
                }
            }
            2 => {
                // verify consistency
                for (&id, &expected_value) in &reference {
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
