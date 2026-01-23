#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::LazyMinHeap;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

// Fuzz stress test with heavy operations and reference validation
//
// Tests behavior under heavy load with reference BinaryHeap implementation
// to ensure min-heap ordering correctness.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
    let mut reference: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    let mut live_keys = std::collections::HashSet::new();

    for chunk in data.chunks(3) {
        if chunk.len() < 2 {
            break;
        }

        let op = chunk[0] % 3;
        let key = u32::from(chunk[1]);
        let score = if chunk.len() > 2 { u32::from(chunk[2]) } else { 0 };

        match op {
            0 => {
                // update
                heap.update(key, score);

                // Update reference: remove old entry if exists, add new one
                reference.retain(|&Reverse((s, k))| k != key);
                reference.push(Reverse((score, key)));
                live_keys.insert(key);
            }
            1 => {
                // pop_best
                let heap_val = heap.pop_best();

                // Find minimum in reference that's still live
                let mut ref_val = None;
                while let Some(Reverse((score, key))) = reference.pop() {
                    if live_keys.contains(&key) {
                        ref_val = Some((key, score));
                        live_keys.remove(&key);
                        break;
                    }
                }

                assert_eq!(heap_val, ref_val);
            }
            2 => {
                // remove
                heap.remove(&key);
                live_keys.remove(&key);
            }
            _ => unreachable!(),
        }

        // Verify length matches live keys
        assert_eq!(heap.len(), live_keys.len());
        assert_eq!(heap.is_empty(), live_keys.is_empty());
    }
});
