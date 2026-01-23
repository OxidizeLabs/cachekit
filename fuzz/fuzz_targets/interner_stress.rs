#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::KeyInterner;
use std::collections::HashMap;

// Fuzz stress test with heavy intern operations and reference validation
//
// Tests behavior under heavy load with reference HashMap implementation
// to ensure handle assignment and bidirectional mapping are correct.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut interner: KeyInterner<u32> = KeyInterner::new();
    let mut reference: HashMap<u32, u64> = HashMap::new();
    let mut reverse: HashMap<u64, u32> = HashMap::new();
    let mut next_handle: u64 = 0;

    for &byte in data {
        let key = u32::from(byte);

        // Intern in both interner and reference
        let handle = interner.intern(&key);

        if let Some(&expected_handle) = reference.get(&key) {
            // Key already exists - should return same handle
            assert_eq!(handle, expected_handle);
        } else {
            // New key - should get next sequential handle
            assert_eq!(handle, next_handle);
            reference.insert(key, handle);
            reverse.insert(handle, key);
            next_handle += 1;
        }

        // Verify length matches
        assert_eq!(interner.len(), reference.len());
        assert_eq!(interner.len() as u64, next_handle);

        // Verify all keys in reference have correct handles
        for (&ref_key, &ref_handle) in &reference {
            assert_eq!(interner.get_handle(&ref_key), Some(ref_handle));
            assert_eq!(interner.resolve(ref_handle), Some(&ref_key));
        }

        // Verify bidirectional mapping
        for (&handle, &expected_key) in &reverse {
            assert_eq!(interner.resolve(handle), Some(&expected_key));
        }
    }

    // Final validation
    assert_eq!(interner.len(), reference.len());
    assert_eq!(interner.len() as u64, next_handle);
});
