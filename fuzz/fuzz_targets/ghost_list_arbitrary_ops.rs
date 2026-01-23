#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::GhostList;

// Fuzz arbitrary operation sequences on GhostList
//
// Tests random sequences of record, remove, contains, clear operations
// to find edge cases and invariant violations in the ghost list.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Use first byte to determine capacity (0-50)
    let capacity = (data[0] as usize) % 51;
    let mut ghost: GhostList<u32> = GhostList::new(capacity);

    let mut idx = 1;
    while idx < data.len() {
        if idx + 1 >= data.len() {
            break;
        }

        let op = data[idx] % 6;
        let key = u32::from(data[idx + 1]);

        match op {
            0 => {
                // Record key
                ghost.record(key);
            }
            1 => {
                // Remove key
                ghost.remove(&key);
            }
            2 => {
                // Check contains (read-only)
                let _ = ghost.contains(&key);
            }
            3 => {
                // Get len (read-only)
                let _ = ghost.len();
            }
            4 => {
                // Clear
                ghost.clear();
            }
            5 => {
                // Clear and shrink
                ghost.clear_shrink();
            }
            _ => unreachable!(),
        }

        // Validate invariants after each operation
        ghost.debug_validate_invariants();

        // Check basic consistency
        assert!(ghost.len() <= ghost.capacity());

        if ghost.is_empty() {
            assert_eq!(ghost.len(), 0);
        } else {
            assert!(ghost.len() > 0);
        }

        // Zero capacity should always be empty
        if capacity == 0 {
            assert!(ghost.is_empty());
            assert_eq!(ghost.len(), 0);
        }

        idx += 2;
    }
});
