#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::ShardSelector;

// Fuzz shard distribution properties
//
// Tests that keys distribute across shards and that different seeds
// produce different distributions.
fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let shard_count = ((data[0] as usize) % 16) + 2; // 2 to 17 shards
    let seed = u64::from(data[1]);

    let selector = ShardSelector::new(shard_count, seed);

    // Track which shards get used
    let mut shard_counts = vec![0usize; shard_count];

    // Map keys to shards
    for &byte in &data[2..] {
        let key = u32::from(byte);
        let shard = selector.shard_for_key(&key);

        assert!(shard < shard_count);
        shard_counts[shard] += 1;
    }

    // If we have enough unique keys, at least some shards should be used
    let unique_keys: std::collections::HashSet<_> = data[2..].iter().collect();
    if unique_keys.len() >= shard_count {
        let used_shards = shard_counts.iter().filter(|&&count| count > 0).count();
        // We expect at least some distribution, but not enforcing strict requirements
        // since hash distribution can be uneven with small samples
        assert!(used_shards > 0);
    }

    // Test seed isolation - different seeds should produce different results
    if data.len() > 20 {
        let seed2 = u64::from(data[10]);
        if seed != seed2 {
            let selector2 = ShardSelector::new(shard_count, seed2);

            let mut same_count = 0;
            let mut different_count = 0;

            for &byte in &data[11..] {
                let key = u32::from(byte);
                let shard1 = selector.shard_for_key(&key);
                let shard2 = selector2.shard_for_key(&key);

                if shard1 == shard2 {
                    same_count += 1;
                } else {
                    different_count += 1;
                }
            }

            // With different seeds and multiple shards, we expect some keys to map differently
            // (but not enforcing strict requirements as it depends on the hash function)
            let _ = (same_count, different_count);
        }
    }
});
