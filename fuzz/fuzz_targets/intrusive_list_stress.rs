#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::IntrusiveList;
use std::collections::VecDeque;

// Fuzz stress test with heavy operations and reference validation
//
// Tests behavior under heavy load with reference VecDeque implementation
// to ensure ordering correctness and consistency.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut list: IntrusiveList<u32> = IntrusiveList::new();
    let mut reference: VecDeque<u32> = VecDeque::new();

    for (idx, &byte) in data.iter().enumerate() {
        let value = u32::from(byte);
        let op = idx % 4;

        match op {
            0 => {
                // push_back
                list.push_back(value);
                reference.push_back(value);
            }
            1 => {
                // push_front
                list.push_front(value);
                reference.push_front(value);
            }
            2 => {
                // pop_front
                let list_val = list.pop_front();
                let ref_val = reference.pop_front();
                assert_eq!(list_val, ref_val);
            }
            3 => {
                // pop_back
                let list_val = list.pop_back();
                let ref_val = reference.pop_back();
                assert_eq!(list_val, ref_val);
            }
            _ => unreachable!(),
        }

        // Verify length matches
        assert_eq!(list.len(), reference.len());

        // Verify front/back match
        assert_eq!(list.front(), reference.front());
        assert_eq!(list.back(), reference.back());

        // Verify is_empty matches
        assert_eq!(list.is_empty(), reference.is_empty());
    }
});
