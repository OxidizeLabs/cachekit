# Benchmark Results

**Quick Links**: [📊 Interactive Charts](charts.html) | [📁 Raw JSON](results.json)

---

## Environment

- **Date**: 2026-02-18T11:08:09.000732604+00:00
- **Commit**: `023146c0ba4a8ade01e046e29f890e8ea896e50a`
- **Branch**: `main`
- **Dirty**: false
- **Rustc**: rustc 1.93.1 (01f6ddf75 2026-02-11)
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
| **Clock-Pro** | 89.14% | 90.65% | 30.55% | 0.00% | 21.09% | 90.38% | 24.77% | 81.54% |
| **FIFO** | 86.74% | 86.16% | 35.49% | 0.00% | 15.58% | 87.58% | 24.64% | 76.77% |
| **Heap-LFU** | 85.32% | 90.67% | 26.51% | 22.52% | 23.42% | 90.03% | 24.67% | 74.81% |
| **LFU** | 89.40% | 90.64% | 25.82% | 0.00% | 20.28% | 91.13% | 24.61% | 82.57% |
| **LIFO** | 63.08% | 90.66% | 26.08% | 24.22% | 29.10% | 90.18% | 24.76% | 80.25% |
| **LRU** | 88.50% | 90.65% | 35.45% | 0.00% | 16.75% | 90.22% | 24.62% | 80.17% |
| **LRU-K** | 89.40% | 90.64% | 25.91% | 0.00% | 20.28% | 91.13% | 24.61% | 82.57% |
| **MFU** | 61.47% | 53.94% | 27.00% | 12.29% | 21.22% | 61.41% | 24.90% | 52.18% |
| **MRU** | 39.71% | 38.07% | 24.90% | 24.22% | 28.90% | 59.53% | 24.74% | 33.10% |
| **NRU** | 88.17% | 90.66% | 26.35% | 24.21% | 29.10% | 90.20% | 24.74% | 80.29% |
| **Random** | 86.84% | 86.34% | 32.85% | 1.82% | 20.72% | 87.70% | 24.66% | 76.98% |
| **S3-FIFO** | 89.48% | 90.63% | 30.35% | 0.00% | 23.18% | 90.95% | 24.73% | 82.49% |
| **SLRU** | 89.40% | 90.63% | 30.95% | 0.00% | 16.92% | 90.89% | 24.75% | 82.57% |

## Throughput (Million ops/sec)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 12.35 | 16.77 | 8.65 |
| **Clock** | 13.96 | 18.01 | 9.29 |
| **Clock-Pro** | 12.86 | 10.30 | 8.30 |
| **FIFO** | 11.62 | 11.92 | 8.07 |
| **Heap-LFU** | 8.02 | 7.46 | 6.05 |
| **LFU** | 8.10 | 8.63 | 5.68 |
| **LIFO** | 14.78 | 21.35 | 9.80 |
| **LRU** | 12.41 | 17.47 | 8.82 |
| **LRU-K** | 11.00 | 12.85 | 7.95 |
| **MFU** | 9.94 | 10.39 | 7.51 |
| **MRU** | 16.09 | 17.71 | 11.60 |
| **NRU** | 13.42 | 0.65 | 7.33 |
| **Random** | 14.46 | 16.84 | 9.55 |
| **S3-FIFO** | 12.75 | 12.74 | 8.38 |
| **SLRU** | 12.42 | 16.69 | 8.66 |

## Latency P99 (nanoseconds)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 80 | 140 | 100 |
| **Clock** | 90 | 120 | 110 |
| **Clock-Pro** | 211 | 311 | 271 |
| **FIFO** | 160 | 200 | 181 |
| **Heap-LFU** | 271 | 331 | 311 |
| **LFU** | 181 | 351 | 200 |
| **LIFO** | 60 | 80 | 71 |
| **LRU** | 80 | 120 | 101 |
| **LRU-K** | 130 | 181 | 140 |
| **MFU** | 240 | 280 | 300 |
| **MRU** | 100 | 130 | 100 |
| **NRU** | 180 | 10990 | 561 |
| **Random** | 100 | 131 | 111 |
| **S3-FIFO** | 131 | 210 | 180 |
| **SLRU** | 80 | 140 | 100 |

## Scan Resistance

| Policy | Baseline | During Scan | Recovery | Score |
|--------|----------|-------------|----------|-------|
| **2Q** | 79.66% | 7.69% | 78.54% | 0.986 |
| **Clock** | 79.66% | 6.90% | 68.54% | 0.860 |
| **Clock-Pro** | 79.66% | 7.69% | 78.78% | 0.989 |
| **FIFO** | 78.36% | 9.22% | 68.55% | 0.875 |
| **Heap-LFU** | 79.21% | 21.89% | 75.79% | 0.957 |
| **LFU** | 79.66% | 7.69% | 78.54% | 0.986 |
| **LIFO** | 79.75% | 19.48% | 80.80% | 1.013 |
| **LRU** | 79.65% | 7.03% | 68.54% | 0.861 |
| **LRU-K** | 79.66% | 7.69% | 78.54% | 0.986 |
| **MFU** | 75.84% | 12.48% | 49.18% | 0.648 |
| **MRU** | 74.88% | 20.67% | 46.96% | 0.627 |
| **NRU** | 79.75% | 19.48% | 80.86% | 1.014 |
| **Random** | 79.55% | 11.72% | 68.53% | 0.861 |
| **S3-FIFO** | 79.66% | 7.69% | 78.82% | 0.989 |
| **SLRU** | 79.66% | 7.69% | 78.54% | 0.986 |

*Score = Recovery/Baseline (1.0 = perfect recovery)*

## Adaptation Speed

| Policy | Stable Hit Rate | Ops to 50% | Ops to 80% |
|--------|-----------------|------------|------------|
| **2Q** | 33.50% | 3072 | 11264 |
| **Clock** | 49.32% | 3072 | 6144 |
| **Clock-Pro** | 36.04% | 8192 | 11264 |
| **FIFO** | 52.25% | 3072 | 6144 |
| **Heap-LFU** | 9.86% | 2048 | 2048 |
| **LFU** | 9.08% | 1024 | 2048 |
| **LIFO** | 0.20% | 5120 | 16384 |
| **LRU** | 49.32% | 3072 | 5120 |
| **LRU-K** | 9.08% | 1024 | 2048 |
| **MFU** | 3.61% | 1024 | 3072 |
| **MRU** | 0.20% | 5120 | 16384 |
| **NRU** | 0.20% | 5120 | 16384 |
| **Random** | 45.02% | 4096 | 8192 |
| **S3-FIFO** | 46.48% | 8192 | 11264 |
| **SLRU** | 27.73% | 3072 | 11264 |

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
