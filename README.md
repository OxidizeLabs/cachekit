# cachekit

[![CI](https://github.com/OxidizeLabs/cachekit/actions/workflows/ci.yml/badge.svg)](https://github.com/OxidizeLabs/cachekit/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/cachekit)](https://crates.io/crates/cachekit)
[![Docs](https://img.shields.io/docsrs/cachekit)](https://docs.rs/cachekit)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue)](https://github.com/OxidizeLabs/cachekit/blob/main/Cargo.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE-APACHE)

**High-performance cache policies and supporting data structures for Rust systems with optional metrics and benchmarks.**

## Why CacheKit

- Pluggable eviction policies with predictable performance characteristics.
- Unified builder API plus direct access for policy-specific operations.
- Optional metrics and benchmarks to validate trade-offs.

## Installation

Add `cachekit` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
cachekit = "0.2.0"
```

From git (bleeding edge):

```toml
[dependencies]
cachekit = { git = "https://github.com/OxidizeLabs/cachekit" }
```

## Feature Flags

| Feature | Enables |
|---------|---------|
| `metrics` | Hit/miss metrics and snapshots |
| `concurrency` | Concurrent wrappers (requires `parking_lot`) |

## Overview

CacheKit is a Rust library that provides:

- High-performance cache replacement policies (e.g., **FIFO**, **LRU**, **LRU-K**, **Clock**, **NRU**, **S3-FIFO**, **SLRU**, **2Q**, and more).
- Supporting data structures and policy primitives for building caches.
- Optional metrics and benchmark harnesses.
- A modular API suitable for embedding in systems where control over caching behavior is critical.

This crate is designed for systems programming, microservices, and performance-critical applications.

## Features

- Policy implementations optimized for performance and predictability.
- Optional integration with metrics collectors (e.g., Prometheus/metrics crates).
- Benchmarks to compare policy performance under real-world workloads.

## Quick Start

### Using the Builder (Recommended)

The `CacheBuilder` provides a unified API for creating caches with any eviction policy:

```rust
use cachekit::builder::{CacheBuilder, CachePolicy};

fn main() {
    // Create an LRU cache with a capacity of 100 entries
    let mut cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);

    // Insert items
    cache.insert(1, "value1".to_string());
    cache.insert(2, "value2".to_string());

    // Retrieve an item
    if let Some(value) = cache.get(&1) {
        println!("Got from cache: {}", value);
    }

    // Check existence and size
    assert!(cache.contains(&1));
    assert_eq!(cache.len(), 2);
}
```

### Available Policies

All policies are available through the unified builder API:

```rust
use cachekit::builder::{CacheBuilder, CachePolicy};

// FIFO - First In, First Out
let fifo = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Fifo);

// LRU - Least Recently Used
let lru = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);

// LRU-K - Scan-resistant LRU (K=2 is common)
let lru_k = CacheBuilder::new(100).build::<u64, String>(CachePolicy::LruK { k: 2 });

// LFU - Least Frequently Used (bucket-based, O(1))
let lfu = CacheBuilder::new(100).build::<u64, String>(
    CachePolicy::Lfu { bucket_hint: None }
);

// HeapLFU - Least Frequently Used (heap-based, O(log n))
let heap_lfu = CacheBuilder::new(100).build::<u64, String>(CachePolicy::HeapLfu);

// 2Q - Two-Queue with configurable probation fraction
let two_q = CacheBuilder::new(100).build::<u64, String>(
    CachePolicy::TwoQ { probation_frac: 0.25 }
);

// S3-FIFO - Scan-resistant FIFO with small + ghost ratios
let s3_fifo = CacheBuilder::new(100).build::<u64, String>(
    CachePolicy::S3Fifo { small_ratio: 0.1, ghost_ratio: 0.9 }
);

// LIFO - Last In, First Out (stack-like eviction)
let lifo = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lifo);

// MFU - Most Frequently Used (evicts hot items)
let mfu = CacheBuilder::new(100).build::<u64, String>(
    CachePolicy::Mfu { bucket_hint: None }
);

// MRU - Most Recently Used (evicts recently accessed)
let mru = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Mru);

// Random - Uniform random eviction
let random = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Random);

// SLRU - Segmented LRU with probationary/protected segments
let slru = CacheBuilder::new(100).build::<u64, String>(
    CachePolicy::Slru { probationary_frac: 0.25 }
);

// Clock - Approximate LRU with reference bits (lower overhead)
let clock = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Clock);

// Clock-PRO - Scan-resistant Clock variant
let clock_pro = CacheBuilder::new(100).build::<u64, String>(CachePolicy::ClockPro);

// NRU - Not Recently Used (simple reference bit tracking)
let nru = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Nru);
```

### Policy Selection Guide

| Policy  | Best For | Eviction Basis |
|---------|----------|----------------|
| FIFO    | Simple, predictable workloads | Insertion order |
| LRU     | Temporal locality | Recency |
| LRU-K   | Scan-resistant workloads | K-th access time |
| LFU     | Stable access patterns | Frequency (O(1)) |
| HeapLFU | Large caches, frequent evictions | Frequency (O(log n)) |
| 2Q      | Mixed workloads | Two-queue promotion |
| S3-FIFO | Scan-heavy workloads | FIFO + ghost history |
| LIFO    | Stack-like caching | Reverse insertion order |
| MFU     | Inverse frequency patterns | Highest frequency |
| MRU     | Anti-recency patterns | Most recent access |
| Random  | Baseline/uniform distribution | Random selection |
| SLRU    | Scan resistance | Segmented LRU |
| Clock   | Low-overhead LRU approximation | Reference bits + hand |
| ClockPro| Scan-resistant Clock variant | Clock + ghost history |
| NRU     | Simple coarse tracking | Reference bits (binary) |

See [Choosing a policy](docs/guides/choosing-a-policy.md) for benchmark-driven guidance.

### Direct Policy Access

For advanced use cases requiring policy-specific operations, use the underlying implementations directly:

```rust
use std::sync::Arc;
use cachekit::policy::lru::LruCore;
use cachekit::traits::{CoreCache, LruCacheTrait};

fn main() {
    // LRU with policy-specific operations
    let mut lru_cache: LruCore<u64, &str> = LruCore::new(100);
    lru_cache.insert(1, Arc::new("value"));

    // Access LRU-specific methods
    if let Some((key, _)) = lru_cache.peek_lru() {
        println!("LRU key: {}", key);
    }
}
```

## Next Steps

- [Quickstart](docs/getting-started/quickstart.md)
- [Integration guide](docs/getting-started/integration.md)
- [Policy overview](docs/policies/README.md)
- [Benchmarks](docs/benchmarks/overview.md)
