# Benchmark Results

**Quick Links**: [📊 Interactive Charts](charts.html) | [📁 Raw JSON](results.json)

---

## Environment

- **Date**: 2026-01-21T18:23:26.347214248+00:00
- **Commit**: `fdc2d5bfff48d361e73803aa8d0e3e9767cb0fd3`
- **Branch**: `main`
- **Dirty**: false
- **Rustc**: rustc 1.92.0 (ded5c06cf 2025-12-08)
- **Host**: x86_64-unknown-linux-gnu
- **CPU**: AMD EPYC 7763 64-Core Processor

## Configuration

- **Capacity**: 4096
- **Universe**: 16384
- **Operations**: 200000
- **Seed**: 42

## Hit Rate Comparison

| Policy | Flash Crowd | HotSet 90/10 | Latest | Scan | Scan Resistance | Scrambled Zipfian | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| **2Q** | 89.36% | 90.63% | 31.55% | 0.00% | 16.04% | 90.71% | 24.78% | 82.37% |
| **Clock** | 88.74% | 90.65% | 34.96% | 0.00% | 17.63% | 90.45% | 24.66% | 80.75% |
| **Heap-LFU** | 85.32% | 90.67% | 26.51% | 22.52% | 23.42% | 90.03% | 24.67% | 74.81% |
| **LFU** | 89.40% | 90.64% | 25.82% | 0.00% | 20.28% | 91.13% | 24.61% | 82.57% |
| **LRU** | 88.50% | 90.65% | 35.45% | 0.00% | 16.75% | 90.22% | 24.62% | 80.17% |
| **LRU-K** | 89.40% | 90.64% | 25.91% | 0.00% | 20.28% | 91.13% | 24.61% | 82.57% |
| **S3-FIFO** | 89.49% | 90.63% | 23.45% | 0.00% | 24.00% | 91.07% | 24.77% | 82.84% |

## Throughput (Million ops/sec)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 11.92 | 16.18 | 8.53 |
| **Clock** | 13.28 | 17.06 | 9.05 |
| **Heap-LFU** | 7.98 | 7.36 | 5.96 |
| **LFU** | 8.06 | 8.51 | 5.75 |
| **LRU** | 12.27 | 17.09 | 8.78 |
| **LRU-K** | 10.86 | 12.64 | 7.91 |
| **S3-FIFO** | 12.20 | 11.85 | 8.19 |

## Latency P99 (nanoseconds)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 80 | 150 | 101 |
| **Clock** | 90 | 130 | 110 |
| **Heap-LFU** | 270 | 341 | 331 |
| **LFU** | 180 | 351 | 210 |
| **LRU** | 80 | 120 | 101 |
| **LRU-K** | 120 | 190 | 140 |
| **S3-FIFO** | 141 | 251 | 200 |

## Scan Resistance

| Policy | Baseline | During Scan | Recovery | Score |
|--------|----------|-------------|----------|-------|
| **2Q** | 79.66% | 7.69% | 78.54% | 0.986 |
| **Clock** | 79.66% | 6.90% | 68.54% | 0.860 |
| **Heap-LFU** | 79.21% | 21.89% | 75.79% | 0.957 |
| **LFU** | 79.66% | 7.69% | 78.54% | 0.986 |
| **LRU** | 79.65% | 7.03% | 68.54% | 0.861 |
| **LRU-K** | 79.66% | 7.69% | 78.54% | 0.986 |
| **S3-FIFO** | 79.66% | 7.69% | 78.82% | 0.989 |

*Score = Recovery/Baseline (1.0 = perfect recovery)*

## Adaptation Speed

| Policy | Stable Hit Rate | Ops to 50% | Ops to 80% |
|--------|-----------------|------------|------------|
| **2Q** | 33.50% | 3072 | 11264 |
| **Clock** | 49.32% | 3072 | 6144 |
| **Heap-LFU** | 9.86% | 2048 | 2048 |
| **LFU** | 9.08% | 1024 | 2048 |
| **LRU** | 49.32% | 3072 | 5120 |
| **LRU-K** | 9.08% | 1024 | 2048 |
| **S3-FIFO** | 44.34% | 8192 | 11264 |

*Lower ops-to-X% is better (faster adaptation)*

## Policy Selection Guide

| Use Case | Recommended Policy | Why |
|----------|-------------------|-----|
| **General purpose, skewed workloads** | LRU, LFU, S3-FIFO | Best hit rates on Zipfian/skewed patterns |
| **Scan-heavy workloads** | S3-FIFO, Heap-LFU | Scan-resistant, protect hot entries |
| **Low latency required** | LRU, Clock | Fastest operations, O(1) overhead |
| **Memory constrained** | LRU, Clock | Minimal metadata overhead |
| **Frequency-aware** | LFU, Heap-LFU, LRU-K | Track access frequency for better decisions |
| **Shifting patterns** | S3-FIFO, 2Q | Adapt to changing access patterns |
| **Multi-access patterns** | 2Q, S3-FIFO | Handle mixed one-hit and frequent items |

---

*Generated from `results.json` (schema v1.0.0)*
