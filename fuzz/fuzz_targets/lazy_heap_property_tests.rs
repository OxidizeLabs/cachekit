#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::LazyMinHeap;

// Fuzz property-based tests for LazyMinHeap
//
// Tests specific invariants and properties:
// - Min-heap ordering (pop returns smallest score)
// - Update idempotency
// - Stale entry skipping
// - Rebuild correctness
// - Clear operation correctness
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let test_type = data[0] % 6;

    match test_type {
        0 => test_min_heap_ordering(&data[1..]),
        1 => test_update_overwrites(&data[1..]),
        2 => test_stale_entries_skipped(&data[1..]),
        3 => test_rebuild_preserves_order(&data[1..]),
        4 => test_remove_consistency(&data[1..]),
        5 => test_clear_operation(&data[1..]),
        _ => unreachable!(),
    }
});

// Property: pop_best returns items in ascending score order
fn test_min_heap_ordering(data: &[u8]) {
    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();

    // Insert key-score pairs
    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let key = u32::from(chunk[0]);
        let score = u32::from(chunk[1]);
        heap.update(key, score);
    }

    // Pop all - scores should be in ascending order
    let mut last_score = None;
    while let Some((_key, score)) = heap.pop_best() {
        if let Some(prev_score) = last_score {
            assert!(score >= prev_score);
        }
        last_score = Some(score);
    }

    assert!(heap.is_empty());
}

// Property: update with new score overwrites old score
fn test_update_overwrites(data: &[u8]) {
    if data.len() < 3 {
        return;
    }

    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
    let key = u32::from(data[0]);

    // Update same key multiple times
    for &byte in &data[1..] {
        let score = u32::from(byte);
        heap.update(key, score);

        // Current score should always match the last update
        assert_eq!(heap.score_of(&key), Some(&score));
        assert_eq!(heap.len(), 1);
    }

    // Pop should return the last score
    if let Some((k, s)) = heap.pop_best() {
        assert_eq!(k, key);
        assert_eq!(s, u32::from(*data.last().unwrap()));
    }
}

// Property: stale entries are skipped during pop_best
fn test_stale_entries_skipped(data: &[u8]) {
    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();

    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let key = u32::from(chunk[0]) % 10; // Limit key range to create updates
        let score = u32::from(chunk[1]);
        heap.update(key, score);
    }

    // After updates, heap may have stale entries
    // But pop_best should skip them and return correct values
    let mut seen_keys = std::collections::HashSet::new();

    while let Some((key, _score)) = heap.pop_best() {
        // Each key should only be popped once
        assert!(!seen_keys.contains(&key));
        seen_keys.insert(key);

        // Score should not be in the scores map anymore
        assert_eq!(heap.score_of(&key), None);
    }
}

// Property: rebuild preserves order and removes stale entries
fn test_rebuild_preserves_order(data: &[u8]) {
    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();

    // Insert with many updates to same keys (creates stale entries)
    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let key = u32::from(chunk[0]) % 5; // Small key range
        let score = u32::from(chunk[1]);
        heap.update(key, score);
    }

    // Collect expected scores before rebuild
    let mut expected: Vec<_> = (0u32..5)
        .filter_map(|k| heap.score_of(&k).map(|&s| (k, s)))
        .collect();
    expected.sort_by_key(|&(_k, s)| s);

    // Rebuild
    let old_len = heap.len();
    heap.rebuild();

    // Length should remain the same
    assert_eq!(heap.len(), old_len);

    // heap_len should now equal len (no stale entries)
    assert_eq!(heap.heap_len(), heap.len());

    // Pop order should match expected
    for (_expected_key, expected_score) in expected {
        let popped = heap.pop_best();
        assert!(popped.is_some());
        let (_key, score) = popped.unwrap();
        assert_eq!(score, expected_score);
        // Key might differ if scores are equal, but score must match
    }
}

// Property: remove makes key unavailable
fn test_remove_consistency(data: &[u8]) {
    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
    let mut keys = Vec::new();

    // Insert keys
    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let key = u32::from(chunk[0]);
        let score = u32::from(chunk[1]);
        heap.update(key, score);
        keys.push(key);
    }

    // Remove each key
    for key in keys {
        let old_len = heap.len();
        let removed = heap.remove(&key);

        if removed.is_some() {
            assert_eq!(heap.len(), old_len - 1);
            assert_eq!(heap.score_of(&key), None);

            // Removed key should not be popped
            // (we can't verify this directly without draining, but score_of should be None)
        }
    }
}

// Property: clear_shrink resets state correctly
fn test_clear_operation(data: &[u8]) {
    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();

    let mut idx = 0;
    while idx < data.len() {
        // Insert some entries
        let insert_count = ((data[idx] as usize) % 10).min(data.len() - idx - 1);

        for i in 0..insert_count {
            if idx + i * 2 + 2 >= data.len() {
                break;
            }
            let key = u32::from(data[idx + i * 2 + 1]);
            let score = u32::from(data[idx + i * 2 + 2]);
            heap.update(key, score);
        }

        idx += insert_count * 2 + 1;

        if idx >= data.len() {
            break;
        }

        // Clear
        heap.clear_shrink();

        // Verify empty state
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.pop_best(), None);

        idx += 1;
    }
}
