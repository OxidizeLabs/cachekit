# cachekit

**High-performance cache policies and tiered caching primitives for Rust systems with optional metrics and benchmarks.**:contentReference[oaicite:1]{index=1}

## Overview

CacheKit is a Rust library that provides:

- High-performance cache replacement policies (e.g., **FIFO**, **LRU**, **LRU-K**).
- Tiered caching primitives to build layered caching strategies.
- Optional metrics and benchmark harnesses.
- A modular API suitable for embedding in systems where control over caching behavior is critical.

This crate is designed for systems programming, microservices, and performance-critical applications.

## Features

- Policy implementations optimized for performance and predictability.
- Backends that support both in-memory and composite cache strategies.
- Optional integration with metrics collectors (e.g., Prometheus/metrics crates).
- Benchmarks to compare policy performance under real-world workloads.
- Idiomatic Rust API with `no_std` compatibility where appropriate.

## Installation

Add `cachekit` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
cachekit = { git = "https://github.com/OxidizeLabs/cachekit" }
```
## example
```rust
use cachekit::policy::lru_k::LRUKCache;

fn main() {
    // Create an LRU cache with a capacity of 100 entries
    let mut cache = LRUKCache::new(2);

    // Insert an item
    cache.insert("key1", "value1");

    // Retrieve an item
    if let Some(value) = cache.get(&"key1") {
        println!("Got from cache: {}", value);
    }
}
```
