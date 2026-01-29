//! Example demonstrating the LIFO (Last In, First Out) cache policy.
//!
//! LIFO evicts the **most recently inserted** item when capacity is reached.
//! This is the opposite of FIFO (which evicts oldest) and is useful for
//! specific patterns like undo buffers or temporary scratch spaces.
//!
//! ⚠️  WARNING: LIFO is a niche policy! Only use when newest items are least needed.
//!
//! Run with: cargo run --example basic_lifo

use cachekit::policy::lifo::LifoCore;

fn main() {
    println!("=== LIFO Cache Example ===\n");

    // Create a LIFO cache with capacity 5
    let mut cache = LifoCore::new(5);

    println!("Created LIFO cache: capacity={}\n", cache.capacity());

    // Insert items 1-5
    for i in 1..=5 {
        cache.insert(i, format!("value-{}", i));
    }
    println!("Inserted keys 1-5 (stack: [1, 2, 3, 4, 5])");
    println!("  len: {}", cache.len());

    // Insert key 6 - LIFO evicts key 5 (most recent!)
    cache.insert(6, "value-6".to_string());
    println!("\nInserted key 6");

    println!("\nAfter inserting key 6:");
    println!(
        "  contains 1? {} (oldest, still in cache)",
        cache.contains(&1)
    );
    println!(
        "  contains 5? {} (was most recent, got evicted!)",
        cache.contains(&5)
    );
    println!("  contains 6? {} (newly inserted)", cache.contains(&6));
    println!("  len: {}", cache.len());

    // Demonstrate the stack behavior
    println!("\n=== Stack Behavior ===\n");

    let mut cache = LifoCore::new(3);

    println!("Insert A, B, C:");
    cache.insert("A", 1);
    cache.insert("B", 2);
    cache.insert("C", 3);
    println!("  Stack (bottom to top): [A, B, C]");
    println!(
        "  All present: A={}, B={}, C={}",
        cache.contains(&"A"),
        cache.contains(&"B"),
        cache.contains(&"C")
    );

    println!("\nInsert D (triggers eviction):");
    cache.insert("D", 4);
    println!("  C evicted (was at top of stack)");
    println!("  Stack: [A, B, D]");
    println!(
        "  A={}, B={}, C={}, D={}",
        cache.contains(&"A"),
        cache.contains(&"B"),
        cache.contains(&"C"),
        cache.contains(&"D")
    );

    println!("\nInsert E:");
    cache.insert("E", 5);
    println!("  D evicted (was at top of stack)");
    println!("  Stack: [A, B, E]");
    println!(
        "  A={}, B={}, D={}, E={}",
        cache.contains(&"A"),
        cache.contains(&"B"),
        cache.contains(&"D"),
        cache.contains(&"E")
    );

    // Show that get doesn't affect eviction
    println!("\n=== Get Doesn't Affect Eviction ===\n");

    let mut cache = LifoCore::new(3);

    cache.insert(1, 10);
    cache.insert(2, 20);
    cache.insert(3, 30);

    println!("Inserted 1, 2, 3");
    println!("Accessing item 1 one hundred times...");
    for _ in 0..100 {
        cache.get(&1);
    }

    println!("Insert item 4:");
    cache.insert(4, 40);

    println!(
        "  contains 1? {} (accessed 100x but position unchanged)",
        cache.contains(&1)
    );
    println!(
        "  contains 3? {} (most recent insert, evicted)",
        cache.contains(&3)
    );
    println!("  contains 4? {} (newly inserted)", cache.contains(&4));
    println!();
    println!("LIFO ignores access patterns - only insertion order matters!");

    // Compare LIFO vs FIFO
    println!("\n=== LIFO vs FIFO Comparison ===\n");

    println!("Key differences:");
    println!("  • LIFO: Evicts from top (newest insertion)");
    println!("  • FIFO: Evicts from bottom (oldest insertion)");
    println!();
    println!("Example with cache capacity 3:");
    println!("  Insert A, B, C → cache: [A, B, C]");
    println!("  Insert D:");
    println!("    • LIFO evicts C (newest) → [A, B, D]");
    println!("    • FIFO evicts A (oldest) → [B, C, D]");

    // Show use case: undo buffer
    println!("\n=== Use Case: Undo Buffer ===\n");

    println!("LIFO is natural for undo/redo operations:");
    println!();
    println!("  1. User does Action A (cached)");
    println!("  2. User does Action B (cached)");
    println!("  3. User does Action C (cached)");
    println!("  4. User hits Undo");
    println!("     → LIFO discards Action C (most recent)");
    println!("     → Perfect! That's what undo should do");
    println!();
    println!("With FIFO, undo would discard Action A instead (wrong!)");

    // Show when LIFO is bad
    println!("\n=== When LIFO Performs Poorly ===\n");

    let mut cache = LifoCore::new(3);

    println!("Typical cache workload:");
    println!("  1. Insert page1");
    cache.insert("page1", "data1");
    println!("  2. Insert page2");
    cache.insert("page2", "data2");
    println!("  3. Insert page3");
    cache.insert("page3", "data3");
    println!("  4. Insert page4 (cache full)");
    cache.insert("page4", "data4");
    println!("     → page3 evicted (most recent before page4)");
    println!("  5. Try to access page3...");
    println!(
        "     contains page3? {} ← MISS! Just evicted what we might need!",
        cache.contains(&"page3")
    );
    println!();
    println!("This is why LIFO is rarely used for general caching.");
    println!("Use LRU, SLRU, or S3-FIFO for typical workloads.");

    // Summary
    println!("\n=== Summary ===\n");
    println!("LIFO characteristics:");
    println!("  ✓ Stack-based eviction (top = newest)");
    println!("  ✓ Opposite of FIFO");
    println!("  ✓ No access pattern tracking");
    println!("  ✓ Good for: undo buffers, temporary scratch space");
    println!("  ✗ Bad for: general-purpose caching");
    println!("  ✗ Counterintuitive: evicts what you just added!");
}

// Expected output:
// === LIFO Cache Example ===
//
// Created LIFO cache: capacity=5
//
// Inserted keys 1-5 (stack: [1, 2, 3, 4, 5])
//   len: 5
//
// Inserted key 6
//
// After inserting key 6:
//   contains 1? true (oldest, still in cache)
//   contains 5? false (was most recent, got evicted!)
//   contains 6? true (newly inserted)
//   len: 5
//
// === Stack Behavior ===
//
// Insert A, B, C:
//   Stack (bottom to top): [A, B, C]
//   All present: A=true, B=true, C=true
//
// Insert D (triggers eviction):
//   C evicted (was at top of stack)
//   Stack: [A, B, D]
//   A=true, B=true, C=false, D=true
//
// Insert E:
//   D evicted (was at top of stack)
//   Stack: [A, B, E]
//   A=true, B=true, D=false, E=true
//
// === Get Doesn't Affect Eviction ===
//
// Inserted 1, 2, 3
// Accessing item 1 one hundred times...
// Insert item 4:
//   contains 1? true (accessed 100x but position unchanged)
//   contains 3? false (most recent insert, evicted)
//   contains 4? true (newly inserted)
//
// LIFO ignores access patterns - only insertion order matters!
//
// === LIFO vs FIFO Comparison ===
//
// Key differences:
//   • LIFO: Evicts from top (newest insertion)
//   • FIFO: Evicts from bottom (oldest insertion)
//
// Example with cache capacity 3:
//   Insert A, B, C → cache: [A, B, C]
//   Insert D:
//     • LIFO evicts C (newest) → [A, B, D]
//     • FIFO evicts A (oldest) → [B, C, D]
//
// === Use Case: Undo Buffer ===
//
// LIFO is natural for undo/redo operations:
//
//   1. User does Action A (cached)
//   2. User does Action B (cached)
//   3. User does Action C (cached)
//   4. User hits Undo
//      → LIFO discards Action C (most recent)
//      → Perfect! That's what undo should do
//
// With FIFO, undo would discard Action A instead (wrong!)
//
// === When LIFO Performs Poorly ===
//
// Typical cache workload:
//   1. Insert page1
//   2. Insert page2
//   3. Insert page3
//   4. Insert page4 (cache full)
//      → page3 evicted (most recent before page4)
//   5. Try to access page3...
//      contains page3? false ← MISS! Just evicted what we might need!
//
// This is why LIFO is rarely used for general caching.
// Use LRU, SLRU, or S3-FIFO for typical workloads.
//
// === Summary ===
//
// LIFO characteristics:
//   ✓ Stack-based eviction (top = newest)
//   ✓ Opposite of FIFO
//   ✓ No access pattern tracking
//   ✓ Good for: undo buffers, temporary scratch space
//   ✗ Bad for: general-purpose caching
//   ✗ Counterintuitive: evicts what you just added!
