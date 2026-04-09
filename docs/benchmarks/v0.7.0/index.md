# Benchmark Results

**Quick Links**: [📊 Interactive Charts](charts.html) | [📁 Raw JSON](results.json)

---

## Environment

- **Date**: 2026-04-09T11:06:55.657222603+00:00
- **Commit**: `400a9d3da5038ff24a9df5dd62a9df2c89656946`
- **Branch**: `HEAD`
- **Dirty**: false
- **Rustc**: rustc 1.94.1 (e408947bf 2026-03-25)
- **Host**: x86_64-unknown-linux-gnu
- **CPU**: AMD EPYC 9V74 80-Core Processor

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
| **MFU** | 41.69% | 37.18% | 24.89% | 18.44% | 23.42% | 59.85% | 24.66% | 31.99% |
| **MRU** | 39.71% | 38.07% | 24.90% | 24.22% | 28.90% | 59.53% | 24.74% | 33.10% |
| **NRU** | 88.17% | 90.66% | 26.35% | 24.21% | 29.10% | 90.20% | 24.74% | 80.29% |
| **Random** | 86.84% | 86.34% | 32.85% | 1.82% | 20.72% | 87.70% | 24.66% | 76.98% |
| **S3-FIFO** | 89.48% | 90.63% | 30.35% | 0.00% | 23.18% | 90.95% | 24.73% | 82.49% |
| **SLRU** | 89.40% | 90.63% | 30.95% | 0.00% | 16.92% | 90.89% | 24.75% | 82.57% |

## Throughput (Million ops/sec)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 10.75 | 15.43 | 7.68 |
| **Clock** | 12.22 | 16.58 | 8.26 |
| **Clock-Pro** | 11.36 | 9.35 | 7.24 |
| **FIFO** | 10.34 | 10.98 | 7.17 |
| **Heap-LFU** | 7.30 | 7.00 | 5.42 |
| **LFU** | 7.04 | 7.97 | 5.07 |
| **LIFO** | 13.04 | 19.29 | 8.61 |
| **LRU** | 9.88 | 15.09 | 7.36 |
| **LRU-K** | 10.05 | 12.07 | 6.41 |
| **MFU** | 9.29 | 9.73 | 7.23 |
| **MRU** | 14.44 | 16.17 | 10.34 |
| **NRU** | 11.70 | 0.64 | 6.38 |
| **Random** | 12.39 | 15.60 | 8.44 |
| **S3-FIFO** | 10.67 | 10.94 | 7.13 |
| **SLRU** | 10.70 | 15.33 | 7.59 |

## Latency P99 (nanoseconds)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 90 | 151 | 111 |
| **Clock** | 91 | 150 | 120 |
| **Clock-Pro** | 220 | 351 | 331 |
| **FIFO** | 170 | 240 | 191 |
| **Heap-LFU** | 290 | 380 | 361 |
| **LFU** | 201 | 381 | 260 |
| **LIFO** | 61 | 90 | 81 |
| **LRU** | 100 | 150 | 150 |
| **LRU-K** | 140 | 200 | 230 |
| **MFU** | 220 | 241 | 251 |
| **MRU** | 101 | 151 | 111 |
| **NRU** | 181 | 11157 | 621 |
| **Random** | 100 | 150 | 120 |
| **S3-FIFO** | 170 | 261 | 250 |
| **SLRU** | 90 | 160 | 111 |

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
| **MFU** | 75.51% | 20.18% | 32.87% | 0.435 |
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
| **MFU** | 3.71% | 1024 | 2048 |
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
