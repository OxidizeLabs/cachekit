#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::GhostList;
use std::collections::VecDeque;

// Fuzz stress test with heavy record operations and LRU validation
//
// Tests behavior under heavy load with reference VecDeque implementation
// to ensure LRU eviction order is correct.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Use first byte to determine capacity (1-20)
    let capacity = ((data[0] as usize) % 20).max(1);
    let mut ghost: GhostList<u32> = GhostList::new(capacity);
    let mut reference: VecDeque<u32> = VecDeque::with_capacity(capacity);

    for &byte in &data[1..] {
        let key = u32::from(byte);

        // Record in both ghost list and reference
        ghost.record(key);

        // Update reference implementation (MRU order)
        if let Some(pos) = reference.iter().position(|&k| k == key) {
            reference.remove(pos);
        } else if reference.len() >= capacity {
            reference.pop_back(); // Remove LRU
        }
        reference.push_front(key); // Add at MRU

        // Validate invariants
        ghost.debug_validate_invariants();

        // Verify length matches
        assert_eq!(ghost.len(), reference.len());
        assert!(ghost.len() <= capacity);

        // Verify all keys in reference are in ghost list
        for &ref_key in &reference {
            assert!(ghost.contains(&ref_key));
        }

        // Verify ghost list doesn't have extra keys
        // (We do this by checking snapshot against reference)
        let snapshot = ghost.debug_snapshot_keys();
        assert_eq!(snapshot.len(), reference.len());
        for snap_key in &snapshot {
            assert!(reference.contains(snap_key));
        }
    }

    // Final validation
    ghost.debug_validate_invariants();
    assert_eq!(ghost.len(), reference.len());
});
