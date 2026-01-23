# Adding New Fuzz Targets

This guide shows how to add new fuzz targets to CacheKit. Thanks to dynamic target discovery, **no CI/CD updates are required** when adding new targets.

## Quick Start

### 1. Create the Fuzz Target

Create a new file in `fuzz/fuzz_targets/`:

```rust
// fuzz/fuzz_targets/my_module_arbitrary_ops.rs
#![no_main]

use libfuzzer_sys::fuzz_target;
use libfuzzer_sys::arbitrary::Arbitrary;

#[derive(Debug, Arbitrary)]
enum Operation {
    Insert(u32, u32),
    Remove(u32),
    Get(u32),
    Clear,
}

fuzz_target!(|ops: Vec<Operation>| {
    let mut my_module = MyModule::new();

    for op in ops {
        match op {
            Operation::Insert(k, v) => { my_module.insert(k, v); }
            Operation::Remove(k) => { my_module.remove(&k); }
            Operation::Get(k) => { let _ = my_module.get(&k); }
            Operation::Clear => { my_module.clear(); }
        }
    }
});
```

### 2. Register in `fuzz/Cargo.toml`

Add a `[[bin]]` section:

```toml
[[bin]]
name = "my_module_arbitrary_ops"
path = "fuzz_targets/my_module_arbitrary_ops.rs"
test = false
doc = false
```

### 3. Test Locally

```bash
cd fuzz
cargo fuzz run my_module_arbitrary_ops -- -max_total_time=60
```

### 4. Commit and Push

```bash
git add fuzz/fuzz_targets/my_module_arbitrary_ops.rs fuzz/Cargo.toml
git commit -m "Add fuzz target for my_module"
git push
```

**That's it!** 🎉 The CI/CD pipeline will automatically:
- Discover your new target
- Run it in PR smoke tests (if named `*_arbitrary_ops`)
- Run it in nightly continuous fuzzing
- Manage its corpus
- Generate coverage reports

## Target Types

Follow this naming convention for optimal CI integration:

### 1. Arbitrary Operations (`*_arbitrary_ops.rs`)

**Purpose**: Test random operation sequences
**CI**: Runs in PR smoke tests (60s) + nightly deep fuzzing (1h)
**Recommended**: Always create this for new modules

```rust
#[derive(Debug, Arbitrary)]
enum Operation {
    // All public operations
}

fuzz_target!(|ops: Vec<Operation>| {
    // Execute operations
});
```

### 2. Stress Testing (`*_stress.rs`)

**Purpose**: Heavy load with reference implementation validation
**CI**: Runs in nightly deep fuzzing only (1h)
**Recommended**: For critical data structures

```rust
fuzz_target!(|data: (Vec<(u32, u32)>, Vec<u32>)| {
    let (inserts, removes) = data;

    // System under test
    let mut sut = MyModule::new();

    // Reference implementation
    let mut reference = std::collections::HashMap::new();

    // Validate equivalence
    for (k, v) in inserts {
        sut.insert(k, v);
        reference.insert(k, v);
    }

    for k in removes {
        assert_eq!(sut.remove(&k), reference.remove(&k));
    }
});
```

### 3. Property Tests (`*_property_tests.rs`)

**Purpose**: Test specific invariants and properties
**CI**: Runs in nightly deep fuzzing only (1h)
**Recommended**: For complex invariants

```rust
fuzz_target!(|ops: Vec<Operation>| {
    let mut module = MyModule::new();

    for op in ops {
        // Execute operation
        match op {
            // ...
        }

        // Validate invariants after each operation
        assert!(module.is_consistent());
        assert_eq!(module.len(), module.count());
        assert!(module.capacity() >= module.len());
    }
});
```

## Best Practices

### 1. Use Arbitrary Types

Leverage `libfuzzer_sys::arbitrary::Arbitrary` for input generation:

```rust
use libfuzzer_sys::arbitrary::Arbitrary;

#[derive(Debug, Arbitrary)]
enum Operation {
    Insert { key: u32, value: String },
    Remove { key: u32 },
}
```

### 2. Add Preconditions

Use early returns to skip invalid inputs:

```rust
fuzz_target!(|data: (usize, Vec<u32>)| {
    let (capacity, items) = data;

    // Skip unreasonable inputs
    if capacity == 0 || capacity > 10_000 {
        return;
    }

    // Test with valid inputs
    let mut module = MyModule::with_capacity(capacity);
    // ...
});
```

### 3. Validate Invariants

Check data structure invariants after operations:

```rust
fuzz_target!(|ops: Vec<Operation>| {
    let mut list = IntrusiveList::new();

    for op in ops {
        // Execute operation
        match op {
            Operation::PushFront(val) => {
                let id = list.push_front(val);

                // Validate invariants
                assert!(list.contains(id));
                assert_eq!(list.front(), Some(id));
                assert_eq!(list.len(), list.iter().count());
            }
            // ...
        }
    }
});
```

### 4. Use Reference Implementations

Compare against standard library types:

```rust
use std::collections::VecDeque;

fuzz_target!(|ops: Vec<Operation>| {
    let mut ghost = GhostList::new(100);
    let mut reference = VecDeque::new();

    for op in ops {
        match op {
            Operation::Record(key) => {
                ghost.record(key);
                reference.push_back(key);
                if reference.len() > 100 {
                    reference.pop_front();
                }
            }
            Operation::Contains(key) => {
                assert_eq!(ghost.contains(&key), reference.contains(&key));
            }
        }
    }
});
```

### 5. Limit Resource Usage

Prevent OOM and timeouts:

```rust
fuzz_target!(|ops: Vec<Operation>| {
    // Limit input size
    if ops.len() > 10_000 {
        return;
    }

    let mut module = MyModule::new();
    let mut item_count = 0;

    for op in ops {
        match op {
            Operation::Insert(k, v) => {
                // Limit total items
                if item_count >= 1_000 {
                    continue;
                }
                module.insert(k, v);
                item_count += 1;
            }
            // ...
        }
    }
});
```

### 6. Test Concurrency (Optional)

For thread-safe data structures:

```rust
use std::sync::Arc;
use std::thread;

fuzz_target!(|data: (Vec<Vec<Operation>>, u8)| {
    let (op_lists, num_threads) = data;
    let num_threads = (num_threads as usize % 8) + 1; // 1-8 threads

    let module = Arc::new(MyThreadSafeModule::new());
    let handles: Vec<_> = op_lists
        .into_iter()
        .take(num_threads)
        .map(|ops| {
            let module = Arc::clone(&module);
            thread::spawn(move || {
                for op in ops {
                    // Execute operations concurrently
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
});
```

## Testing Your Fuzz Target

### Quick Test (60 seconds)

```bash
cd fuzz
cargo fuzz run my_module_arbitrary_ops -- -max_total_time=60
```

### Extended Test (1 hour)

```bash
cd fuzz
cargo fuzz run my_module_arbitrary_ops -- -max_total_time=3600
```

### With Multiple Jobs

```bash
cd fuzz
cargo fuzz run my_module_arbitrary_ops -- -workers=4
```

### Reproduce a Crash

```bash
cd fuzz
cargo fuzz run my_module_arbitrary_ops fuzz/artifacts/my_module_arbitrary_ops/crash-<hash>
```

## Documenting Your Fuzz Target

Add a section to `fuzz/README.md`:

```markdown
### MyModule (3 targets)

#### `my_module_arbitrary_ops`
Tests arbitrary sequences of insert, remove, get, and clear operations.

**Run**:
```bash
cd fuzz
cargo fuzz run my_module_arbitrary_ops
```

#### `my_module_stress`
Stress tests with heavy insertion/deletion and validates against HashMap.

**Run**:
```bash
cd fuzz
cargo fuzz run my_module_stress
```

#### `my_module_property_tests`
Tests specific invariants: length tracking, capacity bounds, etc.

**Run**:
```bash
cd fuzz
cargo fuzz run my_module_property_tests
```
```

## Troubleshooting

### Target Not Discovered

**Problem**: CI doesn't run your new target

**Solution**: Ensure it's registered in `fuzz/Cargo.toml`:
```toml
[[bin]]
name = "my_target"
path = "fuzz_targets/my_target.rs"
test = false
doc = false
```

Verify locally:
```bash
cd fuzz
cargo fuzz list | grep my_target
```

### Target Crashes Immediately

**Problem**: Fuzz target panics on all inputs

**Solution**: Add input validation and early returns:
```rust
fuzz_target!(|data: Vec<u32>| {
    if data.is_empty() || data.len() > 10_000 {
        return; // Skip invalid inputs
    }
    // ... rest of fuzzing logic
});
```

### Target Runs Forever

**Problem**: Operations are too slow or allocate too much

**Solution**: Add resource limits:
```rust
fuzz_target!(|ops: Vec<Operation>| {
    if ops.len() > 1_000 {
        return; // Limit operation count
    }

    let mut module = MyModule::new();
    let mut size = 0;

    for op in ops {
        if size > 10_000 {
            break; // Limit total size
        }
        // ... execute operation
    }
});
```

### Target Finds No Bugs

**Problem**: Fuzzer isn't exploring interesting inputs

**Solution**:
1. Add more diverse operations
2. Use structured input with `Arbitrary`
3. Add assertions for invariants
4. Check coverage with `cargo fuzz coverage`

## Examples

See existing fuzz targets for complete examples:

- **Simple**: `fuzz/fuzz_targets/fixed_history_arbitrary_ops.rs`
- **With Reference**: `fuzz/fuzz_targets/interner_stress.rs`
- **Property Testing**: `fuzz/fuzz_targets/lazy_heap_property_tests.rs`
- **Complex**: `fuzz/fuzz_targets/intrusive_list_arbitrary_ops.rs`

## CI/CD Integration

Your new fuzz target will automatically be:

- ✅ Discovered by `cargo fuzz list`
- ✅ Included in nightly continuous fuzzing (1 hour)
- ✅ Included in PR smoke tests if named `*_arbitrary_ops` (60 seconds)
- ✅ Given its own corpus (cached between runs)
- ✅ Monitored for crashes (auto-creates issues)
- ✅ Included in coverage reports

**No workflow file updates needed!**

## Resources

- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [cargo-fuzz Book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [Arbitrary Trait Documentation](https://docs.rs/arbitrary/latest/arbitrary/)
- [Fuzzing CI/CD Guide](fuzzing-cicd.md)
- [Fuzz Targets Documentation](../../fuzz/README.md)
