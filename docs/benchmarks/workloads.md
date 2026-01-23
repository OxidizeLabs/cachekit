# Workloads

Synthetic workload generators for cache benchmarking. These distributions model
real-world access patterns to evaluate cache policy effectiveness beyond simple
micro-benchmarks.

Located in `benches/common/workload.rs`.

---

## Implemented Workloads

### Uniform

```rust
Workload::Uniform
```

All keys in `[0, universe)` have equal probability. Baseline distribution that
doesn't favor any caching strategy. Useful for measuring raw overhead.

**Models:** Random access patterns, hash table lookups with no locality.

---

### Hotset

```rust
Workload::Hotset { hot_fraction: 0.1, hot_prob: 0.9 }
```

Explicit hot/cold split. A fraction of keys (`hot_fraction`) receives most
accesses (`hot_prob`).

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `hot_fraction` | Fraction of keys that are "hot" | 0.1 (10%) |
| `hot_prob` | Probability of accessing a hot key | 0.8-0.95 |

**Models:** Database tables with popular rows, CDN with viral content.

---

### Scan

```rust
Workload::Scan
```

Sequential access through keys `0, 1, 2, ..., universe-1`, then wraps. Tests
scan resistance - policies that protect the cache from being flushed by
sequential scans.

**Models:** Table scans, batch processing, backup operations.

---

### Zipfian

```rust
Workload::Zipfian { exponent: 1.0 }
```

Power-law distribution where rank-k item has probability proportional to
`1/k^exponent`. Models the natural skew in most real-world access patterns.

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `exponent` | Skew factor (higher = more skewed) | 1.0 (standard Zipf) |

**Models:** Web page popularity, word frequencies, social media engagement.

---

### ScrambledZipfian

```rust
Workload::ScrambledZipfian { exponent: 1.0 }
```

Zipfian distribution with FNV-1a hash applied to keys. **YCSB's default
distribution.** Prevents hardware prefetching from skewing benchmark results
by eliminating sequential locality in hot keys.

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `exponent` | Skew factor | 1.0 |

**Models:** Same as Zipfian but more accurate for benchmarking since it avoids
prefetch effects.

---

### Latest

```rust
Workload::Latest { exponent: 1.0 }
```

Recently inserted keys are more likely to be accessed. Samples an offset from
the most recent insert using Zipfian distribution, then accesses
`insert_counter - offset`.

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `exponent` | How quickly interest falls off | 1.0-1.5 |

**Models:** Social media feeds, news sites, activity logs, chat applications.

---

### ShiftingHotspot

```rust
Workload::ShiftingHotspot { shift_interval: 10_000, hot_fraction: 0.1 }
```

The popular keys change periodically. Every `shift_interval` operations, the
hotspot region moves. Tests how quickly policies adapt to changing access
patterns.

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `shift_interval` | Operations between hotspot shifts | 10,000-100,000 |
| `hot_fraction` | Fraction of keys that are hot | 0.05-0.2 |

**Models:** Trending topics, seasonal products, time-based popularity shifts.

---

### Exponential

```rust
Workload::Exponential { lambda: 0.05 }
```

Exponential decay distribution favoring lower-numbered keys. Popularity drops
exponentially with key distance from 0.

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `lambda` | Decay rate (higher = steeper dropoff) | 0.01-0.1 |

**Models:** Time-series data, event logs, recency-biased access.

---

## Usage

```rust
use crate::common::workload::{Workload, WorkloadSpec, run_hit_rate};

let spec = WorkloadSpec {
    universe: 100_000,          // Key space size
    workload: Workload::ScrambledZipfian { exponent: 1.0 },
    seed: 42,                   // For reproducibility
};

let mut generator = spec.generator();
let hit_rate = run_hit_rate(&mut cache, &mut generator, 1_000_000, |k| Arc::new(k));
println!("Hit rate: {:.2}%", hit_rate.hit_rate() * 100.0);
```

## Choosing a Workload

| Goal | Recommended Workload |
|------|---------------------|
| General benchmarking | `ScrambledZipfian { exponent: 1.0 }` |
| Scan resistance testing | `Scan` or mix with Zipfian |
| Temporal locality | `Latest { exponent: 1.0 }` |
| Adaptation testing | `ShiftingHotspot` |
| Worst-case baseline | `Uniform` |
| Explicit hot/cold | `Hotset` |

---

## Roadmap

Future workloads organized by category. Priority indicated as P1 (high) to P4 (low).

---

### Category 1: Statistical Distributions

Mathematical distributions for modeling different popularity curves.

#### Pareto (P2)

The 80/20 rule distribution. Different tail behavior than Zipf - models cases
where a small percentage of items receive the vast majority of accesses.

```rust
Workload::Pareto { shape: 1.16 }  // shape=1.16 gives 80/20 split
```

**Models:** Wealth distribution, file sizes, resource usage patterns.

#### LogNormal (P3)

Log-normal distribution. Common for quantities that are products of many
independent factors.

```rust
Workload::LogNormal { mu: 0.0, sigma: 1.0 }
```

**Models:** Response times, session lengths, file sizes, latency distributions.

#### Geometric (P4)

Memoryless distribution. Each key has geometrically decreasing probability.

```rust
Workload::Geometric { p: 0.01 }  // p = success probability
```

**Models:** Retry patterns, number of attempts until success, waiting times.

#### ZipfWithCutoff (P3)

Zipfian but caps maximum popularity to prevent extreme skew. More realistic for
bounded systems.

```rust
Workload::ZipfWithCutoff { exponent: 1.0, max_freq_ratio: 100.0 }
```

**Models:** Rate-limited systems, fair-share caches, bounded popularity.

#### Weibull (P3)

Flexible shape parameter allows modeling various failure/reliability patterns.
Generalizes exponential distribution.

```rust
Workload::Weibull { shape: 1.5, scale: 1000.0 }
```

**Models:** Failure times, component lifetimes, reliability analysis.

#### Beta (P4)

Bounded distribution on [0, 1]. Flexible shape for modeling ratios and
probabilities.

```rust
Workload::Beta { alpha: 2.0, beta: 5.0 }
```

**Models:** Conversion rates, click-through ratios, bounded popularity scores.

#### Gamma (P4)

Generalization of exponential. Sum of exponential random variables. Models
waiting times for multiple events.

```rust
Workload::Gamma { shape: 2.0, scale: 1.0 }
```

**Models:** Aggregate latencies, queue waiting times, multi-stage processes.

#### PowerLawCutoff (P3)

Power law with exponential cutoff. More realistic than pure power law - prevents
unbounded popularity in finite systems.

```rust
Workload::PowerLawCutoff { exponent: 1.0, cutoff: 10_000.0 }
```

**Models:** Real-world popularity with natural limits, network degree distributions.

#### DoublePower (P4)

Different exponents for head vs tail. Models systems where very popular items
behave differently from moderately popular ones.

```rust
Workload::DoublePower { head_exponent: 0.8, tail_exponent: 1.5, breakpoint: 100 }
```

**Models:** Celebrity effect, viral vs normal content, tiered popularity.

---

### Category 2: Spatial Locality

Patterns where accessing one key makes nearby keys more likely.

#### Correlated / Stride (P1)

Access to key K makes K+1, K+2, ... more likely. Fundamental pattern in
sequential data processing.

```rust
Workload::Correlated {
    stride: 1,           // Step between correlated accesses
    burst_len: 8,        // Number of sequential accesses in burst
    burst_prob: 0.3,     // Probability of starting a burst
}
```

**Models:** Array traversals, database sequential scans, file system reads,
B-tree leaf scans.

#### Loop (P1)

Repeatedly cycles through a fixed-size working set. Critical test for cache
sizing - behavior changes dramatically when working set exceeds cache.

```rust
Workload::Loop { working_set_size: 1000 }
```

**Models:** Iterative algorithms, hash table resizing, repeated batch jobs.

#### RangeScan (P2)

Random starting point, then sequential burst of N keys. Models range queries
in databases.

```rust
Workload::RangeScan { burst_len: 100 }
```

**Models:** Database range queries, pagination, time-range lookups.

#### RandomWalk (P3)

Current key ± small random delta. Models local exploration patterns where
accesses stay near previous access.

```rust
Workload::RandomWalk { step_size: 10, wrap: true }
```

**Models:** Search exploration, cursor navigation, spatial queries.

#### LevyFlight (P3)

Random walk with heavy-tailed jump sizes. Occasional large jumps mixed with
local exploration. Models foraging and search patterns.

```rust
Workload::LevyFlight { local_prob: 0.9, jump_exponent: 1.5 }
```

**Models:** Search engine crawling, user browsing, exploration algorithms.

---

### Category 3: Multi-Region Patterns

Workloads with multiple distinct hotspots or regions.

#### Bimodal (P2)

Two separate hotspots with configurable split. Models systems with multiple
independent popular categories.

```rust
Workload::Bimodal {
    region_a_fraction: 0.1,
    region_b_fraction: 0.1,
    region_a_prob: 0.4,
    region_b_prob: 0.4,
}
```

**Models:** E-commerce categories, multi-tenant systems, geographic regions.

#### Multimodal (P3)

Generalization of bimodal to N hotspots. Each region has its own size and
access probability.

```rust
Workload::Multimodal {
    regions: vec![
        Region { fraction: 0.05, prob: 0.3 },
        Region { fraction: 0.10, prob: 0.4 },
        Region { fraction: 0.15, prob: 0.2 },
    ]
}
```

**Models:** Complex multi-category systems, hierarchical popularity.

#### Gaussian (P3)

Normal distribution centered on a point. Localized access with gradual falloff
rather than sharp hot/cold boundary.

```rust
Workload::Gaussian { center: 0.5, std_dev: 0.1 }
```

**Models:** Spatial data, location-based access, smooth popularity gradients.

---

### Category 4: Temporal Patterns

Workloads where access patterns change over time.

#### Bursty / SelfSimilar (P1)

Traffic arrives in bursts at multiple time scales. Exhibits long-range
dependence - quiet periods followed by intense bursts.

```rust
Workload::Bursty {
    hurst: 0.8,          // Hurst parameter (0.5=random, 1.0=max correlation)
    base: Box::new(Workload::Zipfian { exponent: 1.0 }),
}
```

**Models:** Network traffic, web requests, API calls, any real-time system.

#### Periodic (P3)

Access pattern repeats with a fixed period. Models cyclical workloads.

```rust
Workload::Periodic {
    period: 1000,        // Operations per cycle
    phases: vec![
        Phase { workload: Workload::Zipfian { exponent: 1.0 }, duration: 800 },
        Phase { workload: Workload::Scan, duration: 200 },
    ]
}
```

**Models:** Daily patterns, scheduled jobs, maintenance windows.

#### ZipfianDrift (P3)

Skew parameter slowly changes over time. Models gradual shifts in access
pattern characteristics.

```rust
Workload::ZipfianDrift {
    start_exponent: 0.8,
    end_exponent: 1.2,
    drift_over: 100_000,  // Operations to complete drift
}
```

**Models:** Evolving user behavior, seasonal changes in popularity distribution.

#### Diurnal (P4)

Different distributions at different simulated times. Models real-world
day/night patterns.

```rust
Workload::Diurnal {
    day_workload: Box::new(Workload::Zipfian { exponent: 1.0 }),
    night_workload: Box::new(Workload::Uniform),
    day_fraction: 0.6,
}
```

**Models:** Business hours vs off-hours, time-zone effects.

#### WorkingSetChurn (P1)

Fixed-size working set that slowly drifts over time. More realistic than
ShiftingHotspot for modeling gradual popularity changes.

```rust
Workload::WorkingSetChurn {
    working_set_size: 1000,
    churn_rate: 0.001,  // Fraction of working set replaced per operation
}
```

**Models:** Cache warmup, gradual content rotation, evolving popular items.

#### FlashCrowd (P1)

Sudden spike in traffic to specific keys. Models viral content or breaking news
scenarios where popularity explodes suddenly.

```rust
Workload::FlashCrowd {
    base: Box::new(Workload::Zipfian { exponent: 1.0 }),
    flash_prob: 0.001,        // Probability of flash event starting
    flash_duration: 1000,     // Operations during flash
    flash_keys: 10,           // Number of keys affected
    flash_intensity: 100.0,   // Multiplier on access probability
}
```

**Models:** Viral content, breaking news, product launches, celebrity posts.

#### ThunderingHerd (P2)

Many simultaneous requests for the same key after a cache miss. Models the
stampede effect when cached item expires.

```rust
Workload::ThunderingHerd {
    base: Box::new(Workload::Zipfian { exponent: 1.0 }),
    herd_size: 100,           // Concurrent requests on miss
    herd_prob: 0.01,          // Probability of herd event
}
```

**Models:** Cache expiration stampedes, cold start scenarios, failover recovery.

#### OnOff (P3)

Alternating active/idle periods per key or globally. Models intermittent
access patterns.

```rust
Workload::OnOff {
    on_duration: 100,
    off_duration: 1000,
    on_workload: Box::new(Workload::Zipfian { exponent: 1.0 }),
}
```

**Models:** Batch jobs, scheduled tasks, user session patterns.

#### MMPP (P4)

Markov-Modulated Poisson Process. Request rate varies according to a hidden
Markov chain. Standard model for network traffic.

```rust
Workload::MMPP {
    states: vec![
        State { rate: 1.0, workload: Workload::Zipfian { exponent: 1.0 } },
        State { rate: 10.0, workload: Workload::Zipfian { exponent: 0.8 } },
    ],
    transition_matrix: [[0.99, 0.01], [0.05, 0.95]],
}
```

**Models:** Network traffic, variable load patterns, hidden state systems.

#### RateVariation (P3)

Request rate changes over time independent of key distribution. Models
varying load levels.

```rust
Workload::RateVariation {
    base: Box::new(Workload::Zipfian { exponent: 1.0 }),
    rate_pattern: RatePattern::Sinusoidal { period: 10_000, amplitude: 0.5 },
}
```

**Models:** Daily traffic patterns, load testing, capacity planning.

---

### Category 5: Composite / Mixed Workloads

Combinations of multiple patterns or operation types.

#### ScanResistance (P1)

Mix of point lookups (Zipfian) and periodic sequential scans. **The key
benchmark for demonstrating S3-FIFO, 2Q, and Clock-PRO advantages over LRU.**

```rust
Workload::ScanResistance {
    scan_fraction: 0.2,      // 20% of ops are scans
    scan_length: 1000,       // Each scan touches 1000 keys
    point_exponent: 1.0,     // Zipfian for point lookups
}
```

**Models:** OLTP + analytics mixed workloads, caches serving batch + interactive.

#### Interleaved (P2)

K independent streams interleaved. Models multi-tenant or multi-user systems
where each stream has its own access pattern.

```rust
Workload::Interleaved {
    streams: 10,
    stream_workload: Box::new(Workload::Zipfian { exponent: 1.0 }),
    stream_universe_fraction: 0.1,  // Each stream uses 10% of key space
}
```

**Models:** Multi-tenant caches, connection pooling, user sessions.

#### ProducerConsumer (P3)

Writes follow one distribution, reads follow another. Models systems where
data ingestion differs from data retrieval.

```rust
Workload::ProducerConsumer {
    write_workload: Box::new(Workload::Latest { exponent: 1.0 }),
    read_workload: Box::new(Workload::Zipfian { exponent: 1.0 }),
    write_fraction: 0.2,
}
```

**Models:** Message queues, event sourcing, write-behind caches.

#### ReadModifyWrite (P2)

Simulates read-then-update patterns. After reading a key, high probability of
writing the same key.

```rust
Workload::ReadModifyWrite {
    base: Box::new(Workload::Zipfian { exponent: 1.0 }),
    rmw_prob: 0.3,  // 30% of reads followed by write
}
```

**Models:** Counters, sessions, shopping carts, any read-update cycle.

#### ReadWriteMix (P2)

Configurable read/write ratio with separate distributions for each.

```rust
Workload::ReadWriteMix {
    read_workload: Box::new(Workload::Zipfian { exponent: 1.0 }),
    write_workload: Box::new(Workload::Uniform),
    read_fraction: 0.95,
}
```

**Models:** General CRUD applications, configurable read-heavy/write-heavy.

#### Mixture (P1)

Weighted combination of N distributions. Meta-workload for creating complex
realistic patterns from simple components.

```rust
Workload::Mixture {
    components: vec![
        (0.7, Box::new(Workload::Zipfian { exponent: 1.0 })),
        (0.2, Box::new(Workload::Scan)),
        (0.1, Box::new(Workload::Uniform)),
    ],
}
```

**Models:** Any complex real-world pattern, configurable benchmarks.

#### TimeVaryingMixture (P2)

Mixture weights change over time. Models gradual shifts between different
access patterns.

```rust
Workload::TimeVaryingMixture {
    components: vec![
        Box::new(Workload::Zipfian { exponent: 1.0 }),
        Box::new(Workload::Scan),
    ],
    weight_function: WeightFunction::Linear {
        start: vec![0.9, 0.1],
        end: vec![0.5, 0.5],
        over_ops: 100_000,
    },
}
```

**Models:** Gradual workload evolution, A/B test transitions, migration patterns.

#### Conditional (P3)

Different distribution based on key properties. Models heterogeneous access
patterns within same key space.

```rust
Workload::Conditional {
    condition: |key| key % 100 == 0,  // Every 100th key
    if_true: Box::new(Workload::Zipfian { exponent: 0.5 }),
    if_false: Box::new(Workload::Zipfian { exponent: 1.5 }),
}
```

**Models:** Tiered data, index vs data pages, metadata vs content.

---

### Category 6: Adversarial / Stress Tests

Workloads designed to stress-test cache policies or expose weaknesses.

#### CacheThrash (P2)

Access pattern sized to exactly overflow cache by a small margin. Tests
worst-case eviction behavior.

```rust
Workload::CacheThrash { overflow_factor: 1.1 }  // 10% larger than cache
```

**Models:** Worst-case scenarios, capacity planning edge cases.

#### Adversarial / AntiLRU (P2)

Cyclic pattern designed to defeat LRU. Accesses keys 0..N in cycle where
N > cache_size. Every access is a miss for pure LRU.

```rust
Workload::Adversarial { cycle_size: cache_size + 1 }
```

**Models:** Sequential scans larger than cache, worst-case for recency policies.

#### AntiLFU (P3)

One-time accesses to many unique keys. Defeats frequency counting by never
accessing keys more than once.

```rust
Workload::AntiLFU { unique_fraction: 0.8 }  // 80% of accesses are unique keys
```

**Models:** Crawlers, one-shot requests, high-cardinality workloads.

#### FlipFlop (P2)

Alternates between two completely disjoint working sets. Tests adaptation speed
when access pattern changes completely.

```rust
Workload::FlipFlop {
    set_a_size: 1000,
    set_b_size: 1000,
    flip_interval: 10_000,
}
```

**Models:** Failover scenarios, A/B deployments, drastic workload changes.

#### Belady / OPT (P4)

Theoretical optimal - always evicts the item used furthest in the future.
Requires pre-computed access trace. Used as upper bound for hit rate.

```rust
Workload::Belady { trace: precomputed_trace }
```

**Models:** Theoretical maximum, used for comparing policy effectiveness.

---

### Category 7: Application-Specific Patterns

Workloads modeling specific application domains.

#### WebCrawler (P3)

Correlated bursts of requests to same domain/prefix. Models web crawling or
API scraping patterns.

```rust
Workload::WebCrawler {
    domain_count: 1000,
    pages_per_domain: 100,
    crawl_depth: 10,
}
```

**Models:** Search engine crawlers, site scrapers, link checkers.

#### DatabaseIndex (P2)

B-tree traversal patterns. Root/internal nodes accessed more frequently than
leaves. Models index lookups.

```rust
Workload::DatabaseIndex {
    fanout: 100,
    levels: 4,
    point_query_prob: 0.8,
}
```

**Models:** B-tree indexes, database buffer pools, file system metadata.

#### LSMCompaction (P4)

Models LSM-tree compaction access patterns. Sequential reads of sorted runs,
writes to new levels.

```rust
Workload::LSMCompaction {
    levels: 4,
    size_ratio: 10,
    compaction_prob: 0.01,
}
```

**Models:** RocksDB, LevelDB, Cassandra compaction workloads.

#### SessionBased (P2)

Keys grouped by session. Sessions have Zipfian popularity, keys within session
accessed with temporal locality.

```rust
Workload::SessionBased {
    session_count: 10_000,
    keys_per_session: 50,
    session_exponent: 1.0,
}
```

**Models:** Web sessions, user activity, shopping carts.

#### GraphTraversal (P3)

BFS/DFS patterns on graph structures. Accesses follow edges, exhibiting
locality based on graph structure.

```rust
Workload::GraphTraversal {
    node_count: 100_000,
    avg_degree: 10,
    traversal: TraversalType::BFS,
}
```

**Models:** Social graph queries, recommendation systems, knowledge graphs.

---

### Category 8: Size-Aware Workloads

For caches with weighted/sized eviction.

#### VariableSize (P3)

Key access distribution combined with size distribution. Tests size-aware
eviction policies.

```rust
Workload::VariableSize {
    access: Box::new(Workload::Zipfian { exponent: 1.0 }),
    size_distribution: SizeDistribution::Pareto { min: 100, shape: 1.5 },
}
```

**Models:** HTTP caches, object stores, CDNs with variable object sizes.

#### SmallHotLargeCold (P3)

Small objects are hot, large objects are cold. Common in practice - popular
items tend to be smaller.

```rust
Workload::SmallHotLargeCold {
    small_size: 1_000,
    large_size: 100_000,
    small_fraction: 0.1,
    small_access_prob: 0.9,
}
```

**Models:** Image thumbnails vs full images, summary vs detail data.

#### CostBenefit (P4)

Each key has an associated cost (fetch latency). Tests GDSF-style policies
that consider cost/size tradeoffs.

```rust
Workload::CostBenefit {
    access: Box::new(Workload::Zipfian { exponent: 1.0 }),
    cost_distribution: CostDistribution::Exponential { mean: 10.0 },
}
```

**Models:** Multi-tier storage, remote fetches with varying latency.

---

### Category 9: Trace Replay

Replay real-world access patterns from trace files.

#### FileTrace (P3)

Replay from trace file. Supports common formats (MSR, Twitter, YCSB).

```rust
Workload::FileTrace {
    path: "traces/twitter_cluster52.trace",
    format: TraceFormat::Twitter,
}
```

**Trace sources:**
- MSR Cambridge traces
- Twitter cache traces
- YCSB generated traces
- CloudPhysics block traces

**Models:** Production workloads, real-world validation.

---

### Category 10: Markov / State-Based

Workloads where next access depends on current state or previous accesses.

#### MarkovChain (P2)

Next key depends on current key with transition probabilities. Models
dependent access patterns with memory.

```rust
Workload::MarkovChain {
    states: 1000,
    transition_matrix: sparse_matrix,  // Or generator function
    initial_state: 0,
}
```

**Models:** User navigation, page sequences, stateful protocols.

#### StackDistance (P2)

Sample directly from LRU stack distance distribution. Gives precise control
over reuse behavior and hit rate.

```rust
Workload::StackDistance {
    distribution: vec![
        (0.3, 1..10),      // 30% within last 10 accesses
        (0.4, 10..100),    // 40% within last 100
        (0.3, 100..10000), // 30% older
    ],
}
```

**Models:** Precise hit rate targeting, stack distance analysis validation.

#### ReuseDistance (P2)

Specify exact reuse distance distribution directly. Number of unique keys
accessed between repeated accesses to same key.

```rust
Workload::ReuseDistance {
    distribution: ReuseDistribution::Exponential { mean: 100.0 },
}
```

**Models:** Working set analysis, cache sizing studies.

#### StateMachine (P3)

General finite state machine with per-state workloads and transition rules.

```rust
Workload::StateMachine {
    states: vec![
        State { name: "browsing", workload: Workload::Zipfian { exponent: 1.0 } },
        State { name: "checkout", workload: Workload::Loop { working_set_size: 20 } },
    ],
    transitions: vec![
        Transition { from: "browsing", to: "checkout", prob: 0.05 },
        Transition { from: "checkout", to: "browsing", prob: 0.8 },
    ],
}
```

**Models:** User journeys, workflow patterns, protocol states.

---

### Category 11: ML/AI Workloads

Patterns specific to machine learning and AI systems.

#### EmbeddingLookup (P2)

Very large embedding tables with extreme power-law access. Models recommendation
systems and NLP embeddings.

```rust
Workload::EmbeddingLookup {
    table_size: 10_000_000,
    exponent: 1.5,           // Steeper than typical Zipf
    batch_size: 64,          // Lookups come in batches
    batch_locality: 0.3,     // Correlation within batch
}
```

**Models:** RecSys embeddings, word2vec, transformer vocabularies.

#### FeatureStore (P2)

Feature access patterns for ML inference. Features grouped by entity with
temporal patterns.

```rust
Workload::FeatureStore {
    entity_count: 1_000_000,
    features_per_entity: 100,
    entity_exponent: 1.0,
    feature_correlation: 0.8,  // Features accessed together
}
```

**Models:** ML feature stores, real-time inference, feature serving.

#### BatchInference (P3)

Batched requests with locality within batch. Models ML serving patterns.

```rust
Workload::BatchInference {
    batch_size: 32,
    intra_batch_locality: 0.5,
    inter_batch_workload: Box::new(Workload::Zipfian { exponent: 1.0 }),
}
```

**Models:** Model serving, batch prediction, GPU inference batching.

#### TrainingDataAccess (P4)

Patterns for training data access. Sequential epochs with shuffling.

```rust
Workload::TrainingDataAccess {
    dataset_size: 1_000_000,
    batch_size: 256,
    shuffle: true,
    prefetch: 10,
}
```

**Models:** ML training data loaders, dataset caching.

---

### Category 12: Hierarchy / Structure

Workloads with hierarchical or structured key relationships.

#### Hierarchical (P2)

Tree-structured keys where parent nodes accessed before children. Models
hierarchical data access.

```rust
Workload::Hierarchical {
    levels: 4,
    fanout: 10,
    level_access_prob: vec![0.4, 0.3, 0.2, 0.1],  // Root accessed most
}
```

**Models:** File systems, URL hierarchies, organizational data.

#### Namespace (P2)

Keys organized in namespaces with locality within namespace. Models
multi-tenant or categorized data.

```rust
Workload::Namespace {
    namespace_count: 100,
    keys_per_namespace: 10_000,
    namespace_exponent: 1.0,
    intra_namespace_locality: 0.8,
}
```

**Models:** Multi-tenant caches, database schemas, API endpoints.

#### PrefixLocality (P3)

Keys with common prefixes accessed together. Models key naming conventions
with hierarchical structure.

```rust
Workload::PrefixLocality {
    prefix_levels: 3,
    prefix_branching: 10,
    locality_decay: 0.7,  // Locality decreases with prefix distance
}
```

**Models:** Key-value stores with hierarchical keys, file paths, URLs.

#### TreeTraversal (P3)

Specific tree traversal patterns (preorder, inorder, postorder, level-order).

```rust
Workload::TreeTraversal {
    tree_size: 10_000,
    fanout: 4,
    traversal: TraversalOrder::LevelOrder,
    subtree_prob: 0.3,  // Probability of full subtree traversal
}
```

**Models:** Index scans, DOM traversal, syntax tree processing.

---

### Category 13: Edge Cases / Degenerate

Extreme or pathological workloads for stress testing.

#### SingleKey (P4)

Only one key ever accessed. 100% hit rate after first access. Tests minimal
overhead.

```rust
Workload::SingleKey { key: 0 }
```

**Models:** Baseline overhead measurement, singleton caches.

#### AllUnique (P4)

Every access is to a new, never-before-seen key. 0% hit rate. Tests miss
handling and eviction overhead.

```rust
Workload::AllUnique
```

**Models:** Worst-case scenarios, cache-hostile workloads, unique ID generation.

#### TwoKeys (P4)

Alternates between exactly two keys. Tests minimal working set handling.

```rust
Workload::TwoKeys { key_a: 0, key_b: 1, prob_a: 0.5 }
```

**Models:** Binary state systems, toggle patterns.

#### RepeatedBurst (P3)

Same key accessed N times consecutively, then moves to next key. Tests
frequency counting and burst handling.

```rust
Workload::RepeatedBurst { burst_size: 100, key_selection: Workload::Uniform }
```

**Models:** Batch processing per key, repeated retries, polling patterns.

#### Pathological (P3)

Specifically designed to expose worst-case behavior for a given policy.
Parameterized by target policy.

```rust
Workload::Pathological { target_policy: Policy::LRU, cache_size: 1000 }
```

**Models:** Policy stress testing, worst-case analysis.

---

### Category 14: Protocol-Specific

Workloads modeling specific protocols and systems.

#### DNS (P3)

DNS query patterns with TTL effects and hierarchical resolution.

```rust
Workload::DNS {
    domain_count: 100_000,
    query_exponent: 1.0,
    ttl_distribution: TTLDistribution::Realistic,
    recursive_prob: 0.1,
}
```

**Models:** DNS caches, resolver caches, CDN DNS.

#### HTTP (P3)

HTTP cache patterns including conditional requests, Vary headers, and
range requests.

```rust
Workload::HTTP {
    url_distribution: Box::new(Workload::Zipfian { exponent: 1.0 }),
    conditional_get_prob: 0.3,
    not_modified_prob: 0.8,
    range_request_prob: 0.1,
}
```

**Models:** HTTP proxy caches, browser caches, CDN edge caches.

#### CDN (P3)

CDN access patterns with geographic distribution, long tail, and edge effects.

```rust
Workload::CDN {
    content_count: 1_000_000,
    popularity_exponent: 1.2,
    geographic_regions: 10,
    regional_locality: 0.7,
    long_tail_fraction: 0.9,
}
```

**Models:** CDN edge caches, video streaming, static asset delivery.

#### Memcached (P2)

Memcached-style patterns with GET/SET/DELETE/INCR operation mix.

```rust
Workload::Memcached {
    key_distribution: Box::new(Workload::Zipfian { exponent: 1.0 }),
    get_fraction: 0.85,
    set_fraction: 0.10,
    delete_fraction: 0.03,
    incr_fraction: 0.02,
}
```

**Models:** Memcached deployments, session stores, rate limiters.

#### Redis (P3)

Redis-style patterns with data structure operations and pub/sub.

```rust
Workload::Redis {
    key_distribution: Box::new(Workload::Zipfian { exponent: 1.0 }),
    string_ops: 0.5,
    hash_ops: 0.2,
    list_ops: 0.15,
    set_ops: 0.1,
    sorted_set_ops: 0.05,
}
```

**Models:** Redis caches, leaderboards, real-time analytics.

---

## Implementation Priority Summary

### P1 - Critical (Implement First)

| Workload | Category | Rationale |
|----------|----------|-----------|
| ScanResistance | Composite | Key differentiator for scan-resistant policies |
| Correlated | Spatial | Fundamental pattern, very common |
| Loop | Spatial | Critical edge case for cache sizing |
| WorkingSetChurn | Temporal | Realistic drift modeling |
| Bursty | Temporal | Real-world traffic is bursty |
| FlashCrowd | Temporal | Critical real-world scenario (viral content) |
| Mixture | Composite | Meta-workload, combines others flexibly |

### P2 - High Value

| Workload | Category | Rationale |
|----------|----------|-----------|
| Pareto | Statistical | Important alternative to Zipf |
| RangeScan | Spatial | Database workloads |
| Bimodal | Multi-Region | Multi-category systems |
| Interleaved | Composite | Multi-tenant scenarios |
| ReadModifyWrite | Composite | Common CRUD pattern |
| ReadWriteMix | Composite | Configurable benchmarks |
| TimeVaryingMixture | Composite | Gradual workload evolution |
| CacheThrash | Adversarial | Stress testing |
| Adversarial | Adversarial | Policy robustness |
| FlipFlop | Adversarial | Adaptation testing |
| ThunderingHerd | Temporal | Concurrency stress test |
| DatabaseIndex | Application | B-tree workloads |
| SessionBased | Application | Web applications |
| MarkovChain | Markov | Models dependent access patterns |
| StackDistance | Markov | Direct control over reuse behavior |
| ReuseDistance | Markov | Precise working set analysis |
| EmbeddingLookup | ML/AI | ML systems are huge cache users |
| FeatureStore | ML/AI | Real-time ML inference |
| Hierarchical | Hierarchy | File systems, URLs |
| Namespace | Hierarchy | Multi-tenant systems |
| Memcached | Protocol | Common deployment pattern |

### P3 - Medium Value

| Workload | Category | Rationale |
|----------|----------|-----------|
| LogNormal | Statistical | Response time modeling |
| ZipfWithCutoff | Statistical | Bounded systems |
| Weibull | Statistical | Reliability modeling |
| PowerLawCutoff | Statistical | Realistic tail behavior |
| Multimodal | Multi-Region | Complex scenarios |
| Gaussian | Multi-Region | Smooth gradients |
| RandomWalk | Spatial | Search/exploration patterns |
| LevyFlight | Spatial | Foraging patterns |
| Periodic | Temporal | Scheduled workloads |
| ZipfianDrift | Temporal | Evolving patterns |
| OnOff | Temporal | Intermittent access |
| RateVariation | Temporal | Variable load levels |
| ProducerConsumer | Composite | Async systems |
| Conditional | Composite | Heterogeneous access |
| AntiLFU | Adversarial | LFU stress test |
| RepeatedBurst | Edge Cases | Burst handling |
| Pathological | Edge Cases | Policy stress testing |
| WebCrawler | Application | Crawler patterns |
| GraphTraversal | Application | Graph databases |
| StateMachine | Markov | User journeys |
| BatchInference | ML/AI | Model serving |
| PrefixLocality | Hierarchy | Key naming conventions |
| TreeTraversal | Hierarchy | Index scans |
| DNS | Protocol | DNS caching |
| HTTP | Protocol | HTTP proxy caches |
| CDN | Protocol | Edge caching |
| Redis | Protocol | Redis patterns |
| VariableSize | Size-Aware | Object caches |
| SmallHotLargeCold | Size-Aware | Common size pattern |
| FileTrace | Trace | Real-world validation |

### P4 - Low Priority

| Workload | Category | Rationale |
|----------|----------|-----------|
| Geometric | Statistical | Niche use cases |
| Beta | Statistical | Bounded distributions |
| Gamma | Statistical | Multi-stage processes |
| DoublePower | Statistical | Complex popularity curves |
| Diurnal | Temporal | Complex to model well |
| MMPP | Temporal | Network modeling, complex |
| Belady | Adversarial | Theoretical only |
| SingleKey | Edge Cases | Trivial baseline |
| AllUnique | Edge Cases | Zero hit rate baseline |
| TwoKeys | Edge Cases | Minimal working set |
| LSMCompaction | Application | Very specific |
| TrainingDataAccess | ML/AI | Training-specific |
| CostBenefit | Size-Aware | Requires cost model |

---

## Workload Count Summary

| Category | Implemented | Roadmap | Total |
|----------|-------------|---------|-------|
| Core (Implemented) | 8 | - | 8 |
| Statistical Distributions | - | 9 | 9 |
| Spatial Locality | - | 5 | 5 |
| Multi-Region Patterns | - | 3 | 3 |
| Temporal Patterns | - | 11 | 11 |
| Composite / Mixed | - | 7 | 7 |
| Adversarial / Stress | - | 5 | 5 |
| Application-Specific | - | 5 | 5 |
| Size-Aware | - | 3 | 3 |
| Trace Replay | - | 1 | 1 |
| Markov / State-Based | - | 4 | 4 |
| ML/AI | - | 4 | 4 |
| Hierarchy / Structure | - | 4 | 4 |
| Edge Cases / Degenerate | - | 5 | 5 |
| Protocol-Specific | - | 5 | 5 |
| **Total** | **8** | **71** | **79** |

---

## References

### Benchmarks & Traces
- [YCSB (Yahoo Cloud Serving Benchmark)](https://github.com/brianfrankcooper/YCSB)
- [MSR Cambridge Traces](http://iotta.snia.org/traces/388)
- [Twitter Cache Traces](https://github.com/twitter/cache-trace)
- [CloudPhysics Block Traces](http://iotta.snia.org/traces/4964)
- [memtier_benchmark](https://github.com/RedisLabs/memtier_benchmark)

### Cache Papers
- [FIFO queues are all you need for cache eviction](https://dl.acm.org/doi/10.1145/3600006.3613147) - S3-FIFO paper
- [ARC: A Self-Tuning, Low Overhead Replacement Cache](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache)
- [TinyLFU: A Highly Efficient Cache Admission Policy](https://dl.acm.org/doi/10.1145/3149371)
- [Adaptive Replacement Cache](https://en.wikipedia.org/wiki/Adaptive_replacement_cache)

### Distributions
- [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law)
- [The Pareto Distribution](https://en.wikipedia.org/wiki/Pareto_distribution)
- [Log-normal Distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
- [Weibull Distribution](https://en.wikipedia.org/wiki/Weibull_distribution)
- [Power Law](https://en.wikipedia.org/wiki/Power_law)

### Traffic Modeling
- [Self-Similar Network Traffic](https://en.wikipedia.org/wiki/Self-similar_process)
- [On the Self-Similar Nature of Ethernet Traffic](https://dl.acm.org/doi/10.1145/167954.166255)
- [Markov-Modulated Poisson Process](https://en.wikipedia.org/wiki/Markov-modulated_Poisson_process)
- [Flash Crowds and DoS Attacks](https://dl.acm.org/doi/10.1145/964723.383073)

### ML/AI Systems
- [Recommendation Systems Embedding Tables](https://arxiv.org/abs/1906.00091)
- [Feature Store Patterns](https://www.featurestore.org/)

### Stack Distance Analysis
- [Stack Distance Analysis](https://en.wikipedia.org/wiki/Stack_distance)
- [Efficient Stack Distance Computation](https://dl.acm.org/doi/10.1145/3524059.3532389)
