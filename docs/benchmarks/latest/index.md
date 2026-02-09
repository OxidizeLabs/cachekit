# Benchmark Results

**Quick Links**: [📊 Interactive Charts](charts.html) | [📁 Raw JSON](results.json)

---

## Environment

- **Date**: 2026-02-09T01:25:06.683106311+00:00
- **Commit**: `b327d82769e3877d3d1f3daf4fd28b15abf28663`
- **Branch**: `main`
- **Dirty**: false
- **Rustc**: rustc 1.93.0 (254b59607 2026-01-19)
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
| **2Q** | 12.17 | 16.64 | 8.71 |
| **Clock** | 13.98 | 17.91 | 9.38 |
| **Clock-Pro** | 12.86 | 10.37 | 8.29 |
| **FIFO** | 11.54 | 11.86 | 8.02 |
| **Heap-LFU** | 8.07 | 7.55 | 6.08 |
| **LFU** | 8.19 | 8.58 | 5.71 |
| **LIFO** | 14.57 | 21.05 | 9.71 |
| **LRU** | 12.35 | 17.34 | 8.94 |
| **LRU-K** | 10.66 | 12.77 | 7.91 |
| **MFU** | 9.97 | 10.41 | 7.55 |
| **MRU** | 15.89 | 17.49 | 11.45 |
| **NRU** | 13.04 | 0.60 | 7.06 |
| **Random** | 14.24 | 16.66 | 9.46 |
| **S3-FIFO** | 12.67 | 12.37 | 8.43 |
| **SLRU** | 12.20 | 16.71 | 8.77 |

## Latency P99 (nanoseconds)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 80 | 140 | 100 |
| **Clock** | 90 | 130 | 110 |
| **Clock-Pro** | 201 | 310 | 261 |
| **FIFO** | 170 | 200 | 190 |
| **Heap-LFU** | 270 | 320 | 300 |
| **LFU** | 181 | 351 | 201 |
| **LIFO** | 60 | 90 | 81 |
| **LRU** | 80 | 120 | 100 |
| **LRU-K** | 130 | 190 | 141 |
| **MFU** | 221 | 271 | 291 |
| **MRU** | 91 | 130 | 100 |
| **NRU** | 190 | 11962 | 621 |
| **Random** | 100 | 130 | 110 |
| **S3-FIFO** | 131 | 210 | 171 |
| **SLRU** | 80 | 141 | 100 |

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
