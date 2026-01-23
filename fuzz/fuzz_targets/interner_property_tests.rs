#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::KeyInterner;

// Fuzz property-based tests for KeyInterner
//
// Tests specific invariants and properties:
// - Monotonic handle assignment
// - Idempotency of intern
// - Bidirectional mapping correctness
// - Clear operation correctness
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let test_type = data[0] % 5;

    match test_type {
        0 => test_monotonic_handles(&data[1..]),
        1 => test_intern_idempotency(&data[1..]),
        2 => test_bidirectional_mapping(&data[1..]),
        3 => test_handle_resolve_consistency(&data[1..]),
        4 => test_clear_operation(&data[1..]),
        _ => unreachable!(),
    }
});

// Property: Handles are assigned monotonically starting from 0
fn test_monotonic_handles(data: &[u8]) {
    let mut interner: KeyInterner<u32> = KeyInterner::new();
    let mut last_handle: Option<u64> = None;

    for &byte in data {
        let key = u32::from(byte);

        if interner.get_handle(&key).is_none() {
            // New key - should get sequential handle
            let handle = interner.intern(&key);

            if let Some(prev) = last_handle {
                assert!(handle > prev);
                assert_eq!(handle, prev + 1);
            } else {
                assert_eq!(handle, 0);
            }

            last_handle = Some(handle);
        }
    }
}

// Property: intern is idempotent - same key always returns same handle
fn test_intern_idempotency(data: &[u8]) {
    let mut interner: KeyInterner<u32> = KeyInterner::new();

    for &byte in data {
        let key = u32::from(byte);

        let handle1 = interner.intern(&key);
        let handle2 = interner.intern(&key);
        let handle3 = interner.intern(&key);

        assert_eq!(handle1, handle2);
        assert_eq!(handle2, handle3);
    }
}

// Property: Bidirectional mapping - intern -> resolve roundtrip
fn test_bidirectional_mapping(data: &[u8]) {
    let mut interner: KeyInterner<u32> = KeyInterner::new();

    for &byte in data {
        let key = u32::from(byte);

        // Intern key
        let handle = interner.intern(&key);

        // Resolve should return the same key
        assert_eq!(interner.resolve(handle), Some(&key));

        // get_handle should also return the same handle
        assert_eq!(interner.get_handle(&key), Some(handle));
    }
}

// Property: get_handle and resolve are consistent
fn test_handle_resolve_consistency(data: &[u8]) {
    let mut interner: KeyInterner<u32> = KeyInterner::new();

    // Intern some keys
    for &byte in data {
        let key = u32::from(byte);
        interner.intern(&key);
    }

    // Verify consistency for all keys
    for &byte in data {
        let key = u32::from(byte);

        if let Some(handle) = interner.get_handle(&key) {
            // If get_handle returns a handle, resolve should return the key
            assert_eq!(interner.resolve(handle), Some(&key));
        }
    }

    // Verify invalid handles return None
    let invalid_handle = interner.len() as u64 + 100;
    assert_eq!(interner.resolve(invalid_handle), None);
}

// Property: clear_shrink resets state correctly
fn test_clear_operation(data: &[u8]) {
    let mut interner: KeyInterner<u32> = KeyInterner::new();

    let mut idx = 0;
    while idx < data.len() {
        // Intern some keys
        let intern_count = ((data[idx] as usize) % 10).min(data.len() - idx - 1);

        let mut handles = Vec::new();
        for i in 0..intern_count {
            if idx + i + 1 >= data.len() {
                break;
            }
            let key = u32::from(data[idx + i + 1]);
            let handle = interner.intern(&key);
            handles.push(handle);
        }

        idx += intern_count + 1;

        if idx >= data.len() {
            break;
        }

        // Clear
        interner.clear_shrink();

        // Verify empty state
        assert!(interner.is_empty());
        assert_eq!(interner.len(), 0);

        // All previous handles should now be invalid
        for handle in handles {
            assert_eq!(interner.resolve(handle), None);
        }

        // All previous keys should not have handles
        for i in 0..intern_count {
            if idx.saturating_sub(intern_count + 1) + i >= data.len() {
                break;
            }
            let key = u32::from(data[idx.saturating_sub(intern_count + 1) + i]);
            assert_eq!(interner.get_handle(&key), None);
        }

        idx += 1;
    }
}
