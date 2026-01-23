#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::SlotArena;

// Fuzz property-based tests for SlotArena
//
// Tests specific invariants:
// - SlotId stability
// - Free slot reuse
// - Length tracking
// - Contains consistency
// - Clear operation correctness
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let test_type = data[0] % 6;

    match test_type {
        0 => test_slot_id_stability(&data[1..]),
        1 => test_free_slot_reuse(&data[1..]),
        2 => test_length_tracking(&data[1..]),
        3 => test_contains_consistency(&data[1..]),
        4 => test_iterator_consistency(&data[1..]),
        5 => test_clear_operation(&data[1..]),
        _ => unreachable!(),
    }
});

// Property: SlotId remains valid until removed
fn test_slot_id_stability(data: &[u8]) {
    let mut arena: SlotArena<u32> = SlotArena::new();

    // Insert values
    let mut ids = Vec::new();
    for &byte in data {
        let value = u32::from(byte);
        let id = arena.insert(value);
        ids.push((id, value));
    }

    // Verify all SlotIds are valid
    for (id, value) in &ids {
        assert_eq!(arena.get(*id), Some(value));
        assert!(arena.contains(*id));
    }

    // Remove some slots
    for (idx, (id, _value)) in ids.iter().enumerate() {
        if idx % 2 == 0 {
            arena.remove(*id);
            assert!(!arena.contains(*id));
            assert_eq!(arena.get(*id), None);
        }
    }

    // Remaining slots should still be valid
    for (idx, (id, value)) in ids.iter().enumerate() {
        if idx % 2 != 0 {
            assert_eq!(arena.get(*id), Some(value));
            assert!(arena.contains(*id));
        }
    }
}

// Property: Freed slots are reused
fn test_free_slot_reuse(data: &[u8]) {
    if data.len() < 4 {
        return;
    }

    let mut arena: SlotArena<u32> = SlotArena::new();

    // Insert some values
    let mut ids = Vec::new();
    for &byte in &data[..data.len() / 2] {
        let value = u32::from(byte);
        let id = arena.insert(value);
        ids.push(id);
    }

    // Remove all slots
    let removed_indices: Vec<_> = ids.iter().map(|id| id.index()).collect();
    for id in &ids {
        arena.remove(*id);
    }

    // Insert new values - should reuse freed slots
    let mut reused_count = 0;
    for &byte in &data[data.len() / 2..] {
        let value = u32::from(byte);
        let id = arena.insert(value);

        if removed_indices.contains(&id.index()) {
            reused_count += 1;
        }
    }

    // At least some slots should have been reused
    if !removed_indices.is_empty() {
        assert!(reused_count > 0);
    }
}

// Property: len() tracks the number of live entries
fn test_length_tracking(data: &[u8]) {
    let mut arena: SlotArena<u32> = SlotArena::new();
    let mut expected_len = 0;

    for &byte in data {
        let value = u32::from(byte);
        let op = byte % 2;

        if op == 0 {
            // insert
            arena.insert(value);
            expected_len += 1;
        } else if !arena.is_empty() {
            // remove
            if let Some(id) = arena.iter_ids().next() {
                arena.remove(id);
                expected_len -= 1;
            }
        }

        assert_eq!(arena.len(), expected_len);
        assert_eq!(arena.is_empty(), expected_len == 0);
    }
}

// Property: contains() is consistent with get()
fn test_contains_consistency(data: &[u8]) {
    let mut arena: SlotArena<u32> = SlotArena::new();
    let mut ids = Vec::new();

    // Insert values
    for &byte in data {
        let value = u32::from(byte);
        let id = arena.insert(value);
        ids.push(id);
    }

    // Verify contains and get are consistent
    for id in &ids {
        let contains = arena.contains(*id);
        let get_result = arena.get(*id);

        if contains {
            assert!(get_result.is_some());
        } else {
            assert!(get_result.is_none());
        }
    }
}

// Property: iter() returns exactly len() items
fn test_iterator_consistency(data: &[u8]) {
    let mut arena: SlotArena<u32> = SlotArena::new();

    for &byte in data {
        let value = u32::from(byte);
        arena.insert(value);
    }

    // iter() should return exactly len() items
    let iter_count = arena.iter().count();
    assert_eq!(iter_count, arena.len());

    // iter_ids() should also return len() items
    let ids_count = arena.iter_ids().count();
    assert_eq!(ids_count, arena.len());
}

// Property: clear_shrink resets state correctly
fn test_clear_operation(data: &[u8]) {
    let mut arena: SlotArena<u32> = SlotArena::new();

    let mut idx = 0;
    while idx < data.len() {
        // Insert some values
        let insert_count = ((data[idx] as usize) % 10).min(data.len() - idx - 1);

        let mut ids = Vec::new();
        for i in 0..insert_count {
            if idx + i + 1 >= data.len() {
                break;
            }
            let value = u32::from(data[idx + i + 1]);
            let id = arena.insert(value);
            ids.push(id);
        }

        idx += insert_count + 1;

        if idx >= data.len() {
            break;
        }

        // Clear
        arena.clear_shrink();

        // Verify empty state
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.iter().count(), 0);

        // All previous ids should be invalid
        for id in ids {
            assert!(!arena.contains(id));
            assert_eq!(arena.get(id), None);
        }

        idx += 1;
    }
}
