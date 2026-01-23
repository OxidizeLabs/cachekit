#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::IntrusiveList;

// Fuzz property-based tests for IntrusiveList
//
// Tests specific invariants and properties:
// - FIFO ordering (push_back + pop_front)
// - LIFO ordering (push_front + pop_front)
// - LRU move to front behavior
// - Remove consistency
// - Clear operation correctness
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let test_type = data[0] % 6;

    match test_type {
        0 => test_fifo_ordering(&data[1..]),
        1 => test_lifo_ordering(&data[1..]),
        2 => test_move_to_front(&data[1..]),
        3 => test_move_to_back(&data[1..]),
        4 => test_remove_consistency(&data[1..]),
        5 => test_clear_operation(&data[1..]),
        _ => unreachable!(),
    }
});

// Property: FIFO ordering - push_back + pop_front preserves order
fn test_fifo_ordering(data: &[u8]) {
    let mut list: IntrusiveList<u32> = IntrusiveList::new();

    // Push all values to back
    for &byte in data {
        let value = u32::from(byte);
        list.push_back(value);
    }

    // Pop from front should return values in same order
    for &byte in data {
        let expected = u32::from(byte);
        let popped = list.pop_front();
        assert_eq!(popped, Some(expected));
    }

    assert!(list.is_empty());
}

// Property: LIFO ordering - push_front + pop_front returns reverse order
fn test_lifo_ordering(data: &[u8]) {
    let mut list: IntrusiveList<u32> = IntrusiveList::new();

    // Push all values to front
    for &byte in data {
        let value = u32::from(byte);
        list.push_front(value);
    }

    // Pop from front should return values in reverse order
    for &byte in data.iter().rev() {
        let expected = u32::from(byte);
        let popped = list.pop_front();
        assert_eq!(popped, Some(expected));
    }

    assert!(list.is_empty());
}

// Property: move_to_front puts element at front (MRU)
fn test_move_to_front(data: &[u8]) {
    if data.is_empty() {
        return;
    }

    let mut list: IntrusiveList<u32> = IntrusiveList::new();
    let mut ids = Vec::new();

    // Push values
    for &byte in data {
        let value = u32::from(byte);
        let id = list.push_back(value);
        ids.push((id, value));
    }

    // Move each element to front and verify
    for (id, value) in ids {
        if list.contains(id) {
            list.move_to_front(id);
            assert_eq!(list.front(), Some(&value));
            assert_eq!(list.front_id(), Some(id));
        }
    }
}

// Property: move_to_back puts element at back (LRU)
fn test_move_to_back(data: &[u8]) {
    if data.is_empty() {
        return;
    }

    let mut list: IntrusiveList<u32> = IntrusiveList::new();
    let mut ids = Vec::new();

    // Push values
    for &byte in data {
        let value = u32::from(byte);
        let id = list.push_front(value);
        ids.push((id, value));
    }

    // Move each element to back and verify
    for (id, value) in ids {
        if list.contains(id) {
            list.move_to_back(id);
            assert_eq!(list.back(), Some(&value));
            assert_eq!(list.back_id(), Some(id));
        }
    }
}

// Property: remove decreases length and makes id invalid
fn test_remove_consistency(data: &[u8]) {
    let mut list: IntrusiveList<u32> = IntrusiveList::new();
    let mut ids = Vec::new();

    // Push values
    for &byte in data {
        let value = u32::from(byte);
        let id = list.push_back(value);
        ids.push((id, value));
    }

    // Remove each element
    for (id, value) in ids {
        if list.contains(id) {
            let old_len = list.len();
            let removed = list.remove(id);

            assert_eq!(removed, Some(value));
            assert_eq!(list.len(), old_len - 1);
            assert!(!list.contains(id));
            assert_eq!(list.get(id), None);
        }
    }

    assert!(list.is_empty());
}

// Property: clear_shrink resets state correctly
fn test_clear_operation(data: &[u8]) {
    let mut list: IntrusiveList<u32> = IntrusiveList::new();
    let mut ids = Vec::new();

    let mut idx = 0;
    while idx < data.len() {
        // Push some values
        let push_count = ((data[idx] as usize) % 10).min(data.len() - idx - 1);

        for i in 0..push_count {
            if idx + i + 1 >= data.len() {
                break;
            }
            let value = u32::from(data[idx + i + 1]);
            let id = list.push_back(value);
            ids.push(id);
        }

        idx += push_count + 1;

        if idx >= data.len() {
            break;
        }

        // Clear
        list.clear_shrink();

        // Verify empty state
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);

        // All previous ids should now be invalid
        for id in &ids {
            assert!(!list.contains(*id));
            assert_eq!(list.get(*id), None);
        }

        ids.clear();
        idx += 1;
    }
}
