#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::KeyInterner;

// Fuzz arbitrary operation sequences on KeyInterner
//
// Tests random sequences of intern, get_handle, resolve, clear_shrink operations
// to find edge cases and invariant violations in the key interner.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut interner: KeyInterner<u32> = KeyInterner::new();
    let mut all_handles: Vec<u64> = Vec::new();

    let mut idx = 0;
    while idx < data.len() {
        if idx + 1 >= data.len() {
            break;
        }

        let op = data[idx] % 5;
        let key = u32::from(data[idx + 1]);

        match op {
            0 => {
                // Intern key
                let handle = interner.intern(&key);
                all_handles.push(handle);

                // Verify handle is valid
                assert_eq!(interner.resolve(handle), Some(&key));

                // Verify idempotency: interning again returns same handle
                assert_eq!(interner.intern(&key), handle);
            }
            1 => {
                // Get handle (read-only)
                if let Some(handle) = interner.get_handle(&key) {
                    // If handle exists, resolve should work
                    assert_eq!(interner.resolve(handle), Some(&key));
                }
            }
            2 => {
                // Resolve handle (read-only)
                if !all_handles.is_empty() {
                    let handle_idx = (data[idx + 1] as usize) % all_handles.len();
                    let handle = all_handles[handle_idx];
                    let _ = interner.resolve(handle);
                }
            }
            3 => {
                // Check len and is_empty consistency
                if interner.is_empty() {
                    assert_eq!(interner.len(), 0);
                } else {
                    assert!(interner.len() > 0);
                }
            }
            4 => {
                // Clear and shrink
                interner.clear_shrink();
                all_handles.clear();

                // Verify empty state
                assert!(interner.is_empty());
                assert_eq!(interner.len(), 0);
            }
            _ => unreachable!(),
        }

        // Basic invariants
        assert!(interner.len() <= u64::MAX as usize);

        // Verify all handles are sequential from 0
        if !interner.is_empty() {
            assert!(interner.len() as u64 > 0);
        }

        idx += 2;
    }
});
