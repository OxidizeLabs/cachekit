# cachekit

[![CI](https://github.com/OxidizeLabs/cachekit/actions/workflows/ci.yml/badge.svg)](https://github.com/OxidizeLabs/cachekit/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/cachekit)](https://crates.io/crates/cachekit)
[![Docs](https://img.shields.io/docsrs/cachekit)](https://docs.rs/cachekit)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue)](https://github.com/OxidizeLabs/cachekit/blob/main/Cargo.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE-APACHE)

**High-performance cache policies and tiered caching primitives for Rust systems with optional metrics and benchmarks.**

## Overview

CacheKit is a Rust library that provides:

- High-performance cache replacement policies (e.g., **FIFO**, **LRU**, **LRU-K**).
- Tiered caching primitives to build layered caching strategies.
- Optional metrics and benchmark harnesses.
- A modular API suitable for embedding in systems where control over caching behavior is critical.

This crate is designed for systems programming, microservices, and performance-critical applications.

## Features

- Policy implementations optimized for performance and predictability.
- Optional integration with metrics collectors (e.g., Prometheus/metrics crates).
- Benchmarks to compare policy performance under real-world workloads.

## Installation

Add `cachekit` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
cachekit = "0.2.0-alpha"
```

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

### Direct Policy Access

For advanced use cases requiring policy-specific operations, use the underlying implementations directly:

```rust
use std::sync::Arc;
use cachekit::policy::lru::LruCore;
use cachekit::traits::{CoreCache, LruCacheTrait};

fn main() {
    let mut cache: LruCore<u64, &str> = LruCore::new(100);
    cache.insert(1, Arc::new("value"));

    // Policy-specific operations
    if let Some((key, _)) = cache.peek_lru() {
        println!("LRU key: {}", key);
    }
}
```
