#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::GhostList;

// Fuzz property-based tests for GhostList
//
// Tests specific invariants and properties:
// - LRU eviction order
// - Promotion to MRU on re-record
// - Capacity bounds
// - Clear operation correctness
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let test_type = data[0] % 5;
    let capacity_byte = data[1];

    match test_type {
        0 => test_lru_eviction_order(&data[2..], capacity_byte),
        1 => test_promotion_to_mru(&data[2..], capacity_byte),
        2 => test_capacity_bounds(&data[2..], capacity_byte),
        3 => test_clear_operations(&data[2..], capacity_byte),
        4 => test_zero_capacity(&data[2..]),
        _ => unreachable!(),
    }
});

// Property: LRU eviction order - oldest keys are evicted first
fn test_lru_eviction_order(data: &[u8], capacity_byte: u8) {
    let capacity = ((capacity_byte as usize) % 10).max(2);
    let mut ghost: GhostList<u32> = GhostList::new(capacity);

    // Record keys in sequence
    let keys: Vec<u32> = data.iter().map(|&b| u32::from(b)).take(20).collect();
    for &key in &keys {
        ghost.record(key);
    }

    // If we recorded more than capacity, verify oldest are gone
    if keys.len() > capacity {
        // First (capacity) unique keys might be evicted
        let mut unique_keys: Vec<u32> = Vec::new();
        for &key in &keys {
            if !unique_keys.contains(&key) {
                unique_keys.push(key);
            }
        }

        // Last 'capacity' unique keys should be present
        let keep_from = unique_keys.len().saturating_sub(capacity);
        for &key in &unique_keys[keep_from..] {
            assert!(ghost.contains(&key));
        }

        // Earlier keys should be evicted (if there were enough unique keys)
        if unique_keys.len() > capacity {
            for &key in &unique_keys[..keep_from.min(3)] {
                assert!(!ghost.contains(&key));
            }
        }
    }

    ghost.debug_validate_invariants();
}

// Property: Re-recording a key promotes it to MRU
fn test_promotion_to_mru(data: &[u8], capacity_byte: u8) {
    let capacity = ((capacity_byte as usize) % 10).max(2);
    let mut ghost: GhostList<u32> = GhostList::new(capacity);

    if data.len() < 3 {
        return;
    }

    // Record initial keys
    for &byte in data.iter().take(capacity) {
        ghost.record(u32::from(byte));
    }

    // Pick a key that's definitely in the list
    if ghost.is_empty() {
        return;
    }

    let snapshot_before = ghost.debug_snapshot_keys();
    if snapshot_before.is_empty() {
        return;
    }

    let promoted_key = snapshot_before[snapshot_before.len() - 1]; // Pick LRU key

    // Promote it
    ghost.record(promoted_key);

    // Verify it's now at MRU position
    let snapshot_after = ghost.debug_snapshot_keys();
    if !snapshot_after.is_empty() {
        assert_eq!(snapshot_after[0], promoted_key);
    }

    // Record one more key to trigger eviction
    let new_key = u32::from(data[data.len() - 1]);
    ghost.record(new_key);

    // Promoted key should still be present (not LRU anymore)
    assert!(ghost.contains(&promoted_key));

    ghost.debug_validate_invariants();
}

// Property: Length never exceeds capacity
fn test_capacity_bounds(data: &[u8], capacity_byte: u8) {
    let capacity = ((capacity_byte as usize) % 20).max(1);
    let mut ghost: GhostList<u32> = GhostList::new(capacity);

    for &byte in data {
        let key = u32::from(byte);
        ghost.record(key);

        assert!(ghost.len() <= capacity);
        ghost.debug_validate_invariants();
    }
}

// Property: Clear operations reset state correctly
fn test_clear_operations(data: &[u8], capacity_byte: u8) {
    let capacity = ((capacity_byte as usize) % 20).max(1);
    let mut ghost: GhostList<u32> = GhostList::new(capacity);

    let mut idx = 0;
    while idx < data.len() {
        // Record some keys
        let record_count = ((data[idx] as usize) % 10).min(data.len() - idx - 1);
        for i in 0..record_count {
            if idx + i + 1 >= data.len() {
                break;
            }
            let key = u32::from(data[idx + i + 1]);
            ghost.record(key);
        }

        idx += record_count + 1;

        if idx >= data.len() {
            break;
        }

        // Choose clear operation
        let clear_op = data[idx] % 2;
        match clear_op {
            0 => ghost.clear(),
            1 => ghost.clear_shrink(),
            _ => unreachable!(),
        }

        // Verify empty state
        assert!(ghost.is_empty());
        assert_eq!(ghost.len(), 0);

        // No keys should be present
        for &byte in data.iter().take(idx) {
            assert!(!ghost.contains(&u32::from(byte)));
        }

        ghost.debug_validate_invariants();

        idx += 1;
    }
}

// Property: Zero capacity ghost list is always empty (no-op)
fn test_zero_capacity(data: &[u8]) {
    let mut ghost: GhostList<u32> = GhostList::new(0);

    for &byte in data {
        let key = u32::from(byte);
        ghost.record(key);

        assert!(ghost.is_empty());
        assert_eq!(ghost.len(), 0);
        assert_eq!(ghost.capacity(), 0);
        assert!(!ghost.contains(&key));

        ghost.debug_validate_invariants();
    }

    // Remove should also be no-op
    for &byte in data {
        let key = u32::from(byte);
        assert!(!ghost.remove(&key));
    }

    assert!(ghost.is_empty());
}
