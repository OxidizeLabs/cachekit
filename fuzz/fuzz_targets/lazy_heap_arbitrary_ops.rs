#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::LazyMinHeap;

// Fuzz arbitrary operation sequences on LazyMinHeap
//
// Tests random sequences of update, remove, pop_best, score_of, rebuild operations.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
    let mut all_keys = Vec::new();

    let mut idx = 0;
    while idx < data.len() {
        if idx + 2 >= data.len() {
            break;
        }

        let op = data[idx] % 8;
        let key = u32::from(data[idx + 1]);
        let score = u32::from(data[idx + 2]);

        match op {
            0 => {
                // update
                heap.update(key, score);
                if !all_keys.contains(&key) {
                    all_keys.push(key);
                }

                // Verify key is now present with correct score
                assert_eq!(heap.score_of(&key), Some(&score));
            }
            1 => {
                // remove
                let old_len = heap.len();
                let removed = heap.remove(&key);

                if removed.is_some() {
                    assert_eq!(heap.len(), old_len - 1);
                    assert_eq!(heap.score_of(&key), None);
                }
            }
            2 => {
                // pop_best
                let old_len = heap.len();
                let popped = heap.pop_best();

                if popped.is_some() {
                    assert_eq!(heap.len(), old_len - 1);
                    let (popped_key, _score) = popped.unwrap();
                    assert_eq!(heap.score_of(&popped_key), None);
                } else {
                    assert_eq!(heap.len(), 0);
                    assert!(heap.is_empty());
                }
            }
            3 => {
                // score_of (read-only)
                let _ = heap.score_of(&key);
            }
            4 => {
                // Check is_empty consistency
                if heap.is_empty() {
                    assert_eq!(heap.len(), 0);
                } else {
                    assert!(!heap.is_empty());
                }
            }
            5 => {
                // rebuild
                let old_len = heap.len();
                heap.rebuild();

                // Length should remain the same
                assert_eq!(heap.len(), old_len);

                // heap_len should now equal len (no stale entries)
                assert_eq!(heap.heap_len(), heap.len());
            }
            6 => {
                // maybe_rebuild with factor 2
                let old_len = heap.len();
                heap.maybe_rebuild(2);
                assert_eq!(heap.len(), old_len);
            }
            7 => {
                // clear_shrink
                heap.clear_shrink();
                all_keys.clear();

                assert!(heap.is_empty());
                assert_eq!(heap.len(), 0);
            }
            _ => unreachable!(),
        }

        // Basic invariants
        if heap.is_empty() {
            assert_eq!(heap.len(), 0);
        }

        idx += 3;
    }
});
