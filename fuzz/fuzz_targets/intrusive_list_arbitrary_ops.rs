#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::IntrusiveList;

// Fuzz arbitrary operation sequences on IntrusiveList
//
// Tests random sequences of push_front, push_back, pop_front, pop_back,
// move_to_front, move_to_back, remove, get, clear operations.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut list: IntrusiveList<u32> = IntrusiveList::new();
    let mut all_ids = Vec::new();

    let mut idx = 0;
    while idx < data.len() {
        if idx + 1 >= data.len() {
            break;
        }

        let op = data[idx] % 11;
        let value = u32::from(data[idx + 1]);

        match op {
            0 => {
                // push_front
                let id = list.push_front(value);
                all_ids.push(id);

                assert_eq!(list.front(), Some(&value));
                assert!(list.contains(id));
                assert_eq!(list.get(id), Some(&value));
            }
            1 => {
                // push_back
                let id = list.push_back(value);
                all_ids.push(id);

                assert_eq!(list.back(), Some(&value));
                assert!(list.contains(id));
                assert_eq!(list.get(id), Some(&value));
            }
            2 => {
                // pop_front
                let old_len = list.len();
                let popped = list.pop_front();

                if popped.is_some() {
                    assert_eq!(list.len(), old_len - 1);
                } else {
                    assert_eq!(list.len(), 0);
                }
            }
            3 => {
                // pop_back
                let old_len = list.len();
                let popped = list.pop_back();

                if popped.is_some() {
                    assert_eq!(list.len(), old_len - 1);
                } else {
                    assert_eq!(list.len(), 0);
                }
            }
            4 => {
                // move_to_front
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];

                    let was_moved = list.move_to_front(id);
                    if was_moved {
                        assert_eq!(list.front_id(), Some(id));
                    }
                }
            }
            5 => {
                // move_to_back
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];

                    let was_moved = list.move_to_back(id);
                    if was_moved {
                        assert_eq!(list.back_id(), Some(id));
                    }
                }
            }
            6 => {
                // remove
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];

                    let old_len = list.len();
                    let removed = list.remove(id);

                    if removed.is_some() {
                        assert_eq!(list.len(), old_len - 1);
                        assert!(!list.contains(id));
                    }
                }
            }
            7 => {
                // get (read-only)
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];
                    let _ = list.get(id);
                }
            }
            8 => {
                // contains (read-only)
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];
                    let _ = list.contains(id);
                }
            }
            9 => {
                // Check is_empty consistency
                if list.is_empty() {
                    assert_eq!(list.len(), 0);
                    assert_eq!(list.front(), None);
                    assert_eq!(list.back(), None);
                } else {
                    assert!(list.len() > 0);
                    assert!(list.front().is_some());
                    assert!(list.back().is_some());
                }
            }
            10 => {
                // clear_shrink
                list.clear_shrink();
                all_ids.clear();

                assert!(list.is_empty());
                assert_eq!(list.len(), 0);
                assert_eq!(list.front(), None);
                assert_eq!(list.back(), None);
            }
            _ => unreachable!(),
        }

        // Basic invariants
        if list.is_empty() {
            assert_eq!(list.len(), 0);
        } else {
            assert!(list.len() > 0);
        }

        idx += 2;
    }
});
