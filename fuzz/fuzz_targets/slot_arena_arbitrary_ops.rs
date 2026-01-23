#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::SlotArena;

// Fuzz arbitrary operation sequences on SlotArena
//
// Tests random sequences of insert, remove, get, get_mut, contains, clear operations.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut arena: SlotArena<u32> = SlotArena::new();
    let mut all_ids = Vec::new();

    let mut idx = 0;
    while idx < data.len() {
        if idx + 1 >= data.len() {
            break;
        }

        let op = data[idx] % 8;
        let value = u32::from(data[idx + 1]);

        match op {
            0 => {
                // insert
                let id = arena.insert(value);
                all_ids.push(id);

                // Verify inserted value can be retrieved
                assert_eq!(arena.get(id), Some(&value));
                assert!(arena.contains(id));
            }
            1 => {
                // remove
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];

                    let old_len = arena.len();
                    let removed = arena.remove(id);

                    if removed.is_some() {
                        assert_eq!(arena.len(), old_len - 1);
                        assert!(!arena.contains(id));
                        assert_eq!(arena.get(id), None);
                    }
                }
            }
            2 => {
                // get (read-only)
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];
                    let _ = arena.get(id);
                }
            }
            3 => {
                // get_mut
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];

                    if let Some(val) = arena.get_mut(id) {
                        *val = value;
                        assert_eq!(arena.get(id), Some(&value));
                    }
                }
            }
            4 => {
                // contains (read-only)
                if !all_ids.is_empty() {
                    let id_idx = (value as usize) % all_ids.len();
                    let id = all_ids[id_idx];
                    let contains = arena.contains(id);

                    // If contains, then get should also work
                    if contains {
                        assert!(arena.get(id).is_some());
                    }
                }
            }
            5 => {
                // Check is_empty consistency
                if arena.is_empty() {
                    assert_eq!(arena.len(), 0);
                } else {
                    assert!(!arena.is_empty());
                }
            }
            6 => {
                // iter (read-only)
                let iter_count = arena.iter().count();
                assert_eq!(iter_count, arena.len());
            }
            7 => {
                // clear_shrink
                arena.clear_shrink();
                all_ids.clear();

                assert!(arena.is_empty());
                assert_eq!(arena.len(), 0);
                assert_eq!(arena.iter().count(), 0);
            }
            _ => unreachable!(),
        }

        // Basic invariants
        if arena.is_empty() {
            assert_eq!(arena.len(), 0);
        }

        idx += 2;
    }
});
