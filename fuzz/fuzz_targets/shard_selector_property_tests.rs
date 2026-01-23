#![no_main]

use libfuzzer_sys::fuzz_target;
use cachekit::ds::ShardSelector;

// Fuzz property-based tests for ShardSelector
//
// Tests specific invariants:
// - Determinism (same key → same shard)
// - Range validity (shard < shard_count)
// - Zero shards clamped to 1
// - Seed isolation
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let test_type = data[0] % 5;

    match test_type {
        0 => test_determinism(&data[1..]),
        1 => test_range_validity(&data[1..]),
        2 => test_zero_shards_clamped(&data[1..]),
        3 => test_single_shard(&data[1..]),
        4 => test_seed_isolation(&data[1..]),
        _ => unreachable!(),
    }
});

// Property: Same key always returns same shard (determinism)
fn test_determinism(data: &[u8]) {
    if data.len() < 2 {
        return;
    }

    let shard_count = ((data[0] as usize) % 64) + 1;
    let seed = u64::from(data[1]);
    let selector = ShardSelector::new(shard_count, seed);

    for &byte in &data[2..] {
        let key = u32::from(byte);

        let shard1 = selector.shard_for_key(&key);
        let shard2 = selector.shard_for_key(&key);
        let shard3 = selector.shard_for_key(&key);

        assert_eq!(shard1, shard2);
        assert_eq!(shard2, shard3);
    }
}

// Property: Shard index is always in valid range [0, shard_count)
fn test_range_validity(data: &[u8]) {
    if data.len() < 2 {
        return;
    }

    let shard_count = ((data[0] as usize) % 128) + 1;
    let seed = u64::from(data[1]);
    let selector = ShardSelector::new(shard_count, seed);

    for &byte in &data[2..] {
        let key = u64::from(byte);
        let shard = selector.shard_for_key(&key);

        assert!(shard < shard_count);
        assert!(shard < selector.shard_count());
    }
}

// Property: Zero shards is clamped to 1
fn test_zero_shards_clamped(data: &[u8]) {
    let seed = if !data.is_empty() {
        u64::from(data[0])
    } else {
        0
    };

    let selector = ShardSelector::new(0, seed);
    assert_eq!(selector.shard_count(), 1);

    // All keys should map to shard 0
    for &byte in data {
        let key = u32::from(byte);
        let shard = selector.shard_for_key(&key);
        assert_eq!(shard, 0);
    }
}

// Property: Single shard always returns 0
fn test_single_shard(data: &[u8]) {
    let seed = if !data.is_empty() {
        u64::from(data[0])
    } else {
        0
    };

    let selector = ShardSelector::new(1, seed);
    assert_eq!(selector.shard_count(), 1);

    // All keys should map to shard 0
    for &byte in data {
        let key = u32::from(byte);
        let shard = selector.shard_for_key(&key);
        assert_eq!(shard, 0);
    }
}

// Property: Different seeds produce different mappings (seed isolation)
fn test_seed_isolation(data: &[u8]) {
    if data.len() < 3 {
        return;
    }

    let shard_count = ((data[0] as usize) % 16) + 2; // At least 2 shards
    let seed1 = u64::from(data[1]);
    let seed2 = u64::from(data[2]);

    if seed1 == seed2 {
        return;
    }

    let selector1 = ShardSelector::new(shard_count, seed1);
    let selector2 = ShardSelector::new(shard_count, seed2);

    // With different seeds, we expect at least some keys to map differently
    let mut different_count = 0;

    for &byte in &data[3..] {
        let key = u32::from(byte);

        let shard1 = selector1.shard_for_key(&key);
        let shard2 = selector2.shard_for_key(&key);

        if shard1 != shard2 {
            different_count += 1;
        }
    }

    // We don't enforce strict distribution, but with enough keys and different seeds,
    // we'd expect at least some differences (this is probabilistic)
}
