//! Basic example demonstrating MFU (Most Frequently Used) cache behavior.
//!
//! MFU evicts the entry with the **highest** frequency, which is counterintuitive
//! for most caching scenarios but useful in specific cases.

use cachekit::policy::mfu::MfuCore;

fn main() {
    println!("=== MFU Cache Example ===\n");

    // Create MFU cache with capacity 3
    let mut cache = MfuCore::new(3);

    println!("Created MFU cache: capacity=3\n");

    // Insert initial items
    cache.insert(1, 100);
    cache.insert(2, 200);
    cache.insert(3, 300);

    println!("Inserted keys 1, 2, 3 (all have frequency 1)");
    println!("  freq 1: {:?}", cache.frequency(&1));
    println!("  freq 2: {:?}", cache.frequency(&2));
    println!("  freq 3: {:?}", cache.frequency(&3));
    println!("  len: {}\n", cache.len());

    // Access item 1 many times
    println!("Accessing item 1 ten times...");
    for _ in 0..10 {
        cache.get(&1);
    }

    println!("After accessing item 1:");
    println!("  freq 1: {:?} (highest!)", cache.frequency(&1));
    println!("  freq 2: {:?}", cache.frequency(&2));
    println!("  freq 3: {:?}\n", cache.frequency(&3));

    // Insert new item, which triggers MFU eviction
    println!("Inserting key 4 (cache full, triggers eviction)...\n");
    cache.insert(4, 400);

    println!("After inserting key 4:");
    println!(
        "  contains 1? {} (highest freq, evicted!)",
        cache.contains(&1)
    );
    println!("  contains 2? {} (low freq, kept)", cache.contains(&2));
    println!("  contains 3? {} (low freq, kept)", cache.contains(&3));
    println!("  contains 4? {} (newly inserted)", cache.contains(&4));
    println!("  len: {}\n", cache.len());

    println!("=== MFU Eviction Behavior ===\n");

    // Demonstrate MFU-specific behavior
    let mut cache2 = MfuCore::new(3);
    cache2.insert("A", "alpha");
    cache2.insert("B", "beta");
    cache2.insert("C", "gamma");

    println!("Created new cache with keys A, B, C");
    println!("  All have frequency 1\n");

    // Create different frequency patterns
    println!("Accessing keys to create frequency distribution:");
    for _ in 0..50 {
        cache2.get(&"A"); // Very high frequency
    }
    for _ in 0..5 {
        cache2.get(&"B"); // Medium frequency
    }
    // C stays at frequency 1 (low)

    println!(
        "  freq A: {:?} (highest - burst activity)",
        cache2.frequency(&"A")
    );
    println!("  freq B: {:?}", cache2.frequency(&"B"));
    println!("  freq C: {:?} (lowest)\n", cache2.frequency(&"C"));

    println!("Inserting key D...");
    cache2.insert("D", "delta");

    println!("\nAfter insertion:");
    println!(
        "  contains A? {} ← EVICTED (highest frequency!)",
        cache2.contains(&"A")
    );
    println!(
        "  contains B? {} (medium freq, kept)",
        cache2.contains(&"B")
    );
    println!(
        "  contains C? {} (lowest freq, kept!)",
        cache2.contains(&"C")
    );
    println!("  contains D? {} (newly inserted)", cache2.contains(&"D"));
    println!();
    println!("MFU evicted the burst item (A) despite being heavily used!");
    println!("This is opposite of LFU behavior.\n");

    println!("=== MFU vs LRU Comparison ===\n");

    let mut mfu_cache = MfuCore::new(3);
    mfu_cache.insert(1, 100);
    mfu_cache.insert(2, 200);
    mfu_cache.insert(3, 300);

    // Access pattern: hot item
    for _ in 0..20 {
        mfu_cache.get(&1);
    }

    println!("Workload: Item 1 accessed 20 times (hot), others once");
    println!("  freq 1: {:?}", mfu_cache.frequency(&1));
    println!("  freq 2: {:?}", mfu_cache.frequency(&2));
    println!("  freq 3: {:?}\n", mfu_cache.frequency(&3));

    mfu_cache.insert(4, 400);

    println!("Insert new item:");
    println!("  MFU evicts: item 1 (highest freq = 21)");
    println!("  LRU would evict: item 2 or 3 (least recently used)");
    println!();
    println!("MFU behavior:");
    println!(
        "  contains 1? {} ← Hot item evicted!",
        mfu_cache.contains(&1)
    );
    println!("  contains 2? {}", mfu_cache.contains(&2));
    println!("  contains 3? {}", mfu_cache.contains(&3));
    println!("  contains 4? {}", mfu_cache.contains(&4));
    println!();
    println!("MFU sacrifices hot items - usually NOT what you want!\n");

    println!("=== Burst Detection Use Case ===\n");

    let mut burst_cache = MfuCore::new(4);

    // Simulate ongoing workload
    println!("Ongoing workload: items A, B, C, D accessed normally");
    burst_cache.insert("A", "data_a");
    burst_cache.insert("B", "data_b");
    burst_cache.insert("C", "data_c");
    burst_cache.insert("D", "data_d");

    for _ in 0..3 {
        burst_cache.get(&"A");
        burst_cache.get(&"B");
        burst_cache.get(&"C");
        burst_cache.get(&"D");
    }

    println!("  freq A: {:?}", burst_cache.frequency(&"A"));
    println!("  freq B: {:?}", burst_cache.frequency(&"B"));
    println!("  freq C: {:?}", burst_cache.frequency(&"C"));
    println!("  freq D: {:?}\n", burst_cache.frequency(&"D"));

    // Sudden burst on one item (e.g., one-time scan)
    println!("Sudden burst: item A accessed 100 times (one-time scan)");
    for _ in 0..100 {
        burst_cache.get(&"A");
    }

    println!("  freq A: {:?} (burst!)\n", burst_cache.frequency(&"A"));

    println!("Insert new item E...");
    burst_cache.insert("E", "data_e");

    println!("\nResult:");
    println!(
        "  contains A? {} ← Burst item evicted",
        burst_cache.contains(&"A")
    );
    println!(
        "  contains B? {} (normal activity, kept)",
        burst_cache.contains(&"B")
    );
    println!(
        "  contains C? {} (normal activity, kept)",
        burst_cache.contains(&"C")
    );
    println!(
        "  contains D? {} (normal activity, kept)",
        burst_cache.contains(&"D")
    );
    println!("  contains E? {} (new item)", burst_cache.contains(&"E"));
    println!();
    println!("MFU detected and evicted the burst activity!");
    println!("This can be useful for anti-scan protection.\n");

    println!("=== peek_mfu and pop_mfu ===\n");

    let mut demo_cache = MfuCore::new(3);
    demo_cache.insert("x", 10);
    demo_cache.insert("y", 20);
    demo_cache.insert("z", 30);

    // Create frequency gradient
    for _ in 0..15 {
        demo_cache.get(&"x");
    }
    for _ in 0..5 {
        demo_cache.get(&"y");
    }

    println!("Cache state:");
    println!("  freq x: {:?}", demo_cache.frequency(&"x"));
    println!("  freq y: {:?}", demo_cache.frequency(&"y"));
    println!("  freq z: {:?}\n", demo_cache.frequency(&"z"));

    // Peek at MFU item
    if let Some((key, value)) = demo_cache.peek_mfu() {
        println!("peek_mfu:");
        println!(
            "  key: {:?}, value: {:?}, freq: {:?}",
            key,
            value,
            demo_cache.frequency(key)
        );
        println!("  (item not removed)\n");
    }

    assert_eq!(demo_cache.len(), 3);

    // Pop MFU item
    if let Some((key, value)) = demo_cache.pop_mfu() {
        println!("pop_mfu:");
        println!("  key: {:?}, value: {:?}", key, value);
        println!("  (item removed)\n");
    }

    println!("After pop_mfu:");
    println!("  len: {}", demo_cache.len());
    println!("  contains x? {}\n", demo_cache.contains(&"x"));

    println!("=== When NOT to Use MFU ===\n");

    let mut bad_cache = MfuCore::new(3);
    bad_cache.insert("page1", "content1");
    bad_cache.insert("page2", "content2");
    bad_cache.insert("page3", "content3");

    println!("Typical workload with temporal locality:");
    println!("  1. Access page1 (user viewing)");
    for _ in 0..5 {
        bad_cache.get(&"page1");
    }

    println!("  2. Access page2 (related content)");
    for _ in 0..3 {
        bad_cache.get(&"page2");
    }

    println!("  3. Insert page4 (new content)\n");
    bad_cache.insert("page4", "content4");

    println!("Result:");
    println!(
        "  contains page1? {} ← MISS! Just used heavily!",
        bad_cache.contains(&"page1")
    );
    println!("  contains page2? {}", bad_cache.contains(&"page2"));
    println!(
        "  contains page3? {} ← Kept despite not being used!",
        bad_cache.contains(&"page3")
    );
    println!("  contains page4? {}", bad_cache.contains(&"page4"));
    println!();
    println!("MFU evicted the hot items we just used!");
    println!("This is terrible for typical caching workloads.\n");

    println!("=== Summary ===\n");

    println!("MFU characteristics:");
    println!("  ✓ Evicts highest frequency items");
    println!("  ✓ O(log n) insert/get operations");
    println!("  ✓ Useful for burst detection");
    println!("  ✓ Can serve as anti-scan protection");
    println!("  ✓ Good for baseline comparisons");
    println!();
    println!("  ✗ Poor for general-purpose caching");
    println!("  ✗ Evicts hot items (opposite of desired behavior)");
    println!("  ✗ Ignores temporal locality");
    println!("  ✗ Higher memory overhead than simple policies");
    println!();
    println!("For typical caching, use LFU, LRU, S3-FIFO, or SLRU instead!");
    println!("MFU is primarily useful for specialized scenarios or research.");
}
