#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::ShardSelector;

// Fuzz arbitrary shard selection operations
//
// Tests determinism and range validity for various shard counts and keys.
fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    // Extract shard count and seed from first bytes
    let shard_count = ((data[0] as usize) % 64) + 1; // 1 to 64 shards
    let seed = u64::from(data[1]);

    let selector = ShardSelector::new(shard_count, seed);

    // Verify shard_count is at least 1
    assert!(selector.shard_count() >= 1);
    assert_eq!(selector.shard_count(), shard_count);

    // Test shard selection for various keys
    for &byte in &data[2..] {
        let key = u32::from(byte);

        // Get shard for this key
        let shard = selector.shard_for_key(&key);

        // Verify shard is in valid range
        assert!(shard < selector.shard_count());

        // Verify determinism - same key returns same shard
        let shard2 = selector.shard_for_key(&key);
        assert_eq!(shard, shard2);
    }

    // Test with string keys
    for chunk in data[2..].chunks(2) {
        if chunk.is_empty() {
            break;
        }
        let key = format!("key_{}", chunk[0]);
        let shard = selector.shard_for_key(&key);

        assert!(shard < selector.shard_count());
        assert_eq!(shard, selector.shard_for_key(&key));
    }
});
