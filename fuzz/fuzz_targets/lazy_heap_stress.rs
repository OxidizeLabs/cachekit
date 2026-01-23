#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::LazyMinHeap;
use std::collections::{BTreeMap, HashMap};

// Fuzz stress test with heavy operations and reference validation
//
// Tests behavior under heavy load with reference BinaryHeap implementation
// to ensure min-heap ordering correctness.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
    let mut reference: BTreeMap<(u32, u64), u32> = BTreeMap::new();
    let mut ref_by_key: HashMap<u32, (u32, u64)> = HashMap::new();
    let mut ref_seq: u64 = 0;

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
                if let Some((old_score, old_seq)) = ref_by_key.remove(&key) {
                    reference.remove(&(old_score, old_seq));
                }
                let seq = ref_seq;
                ref_seq = ref_seq.wrapping_add(1);
                reference.insert((score, seq), key);
                ref_by_key.insert(key, (score, seq));
            }
            1 => {
                // pop_best
                let heap_val = heap.pop_best();

                // Find minimum in reference (score, then seq)
                let ref_val = if let Some((&(score, seq), &key)) = reference.iter().next() {
                    reference.remove(&(score, seq));
                    ref_by_key.remove(&key);
                    Some((key, score))
                } else {
                    None
                };

                assert_eq!(heap_val, ref_val);
            }
            2 => {
                // remove
                heap.remove(&key);
                if let Some((score, seq)) = ref_by_key.remove(&key) {
                    reference.remove(&(score, seq));
                }
            }
            _ => unreachable!(),
        }

        // Verify length matches live keys
        assert_eq!(heap.len(), ref_by_key.len());
        assert_eq!(heap.is_empty(), ref_by_key.is_empty());
    }
});
