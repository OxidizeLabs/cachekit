# Benchmark Results

**Quick Links**: [Interactive Charts](charts.html) | [Raw JSON](results.json)

---

## Environment

- **Date**: 2026-04-27T02:02:03.446581168+00:00
- **Commit**: `e313874e12b4da8186c219190e96533f8826e246`
- **Branch**: `main`
- **Dirty**: false
- **Rustc**: rustc 1.95.0 (59807616e 2026-04-14)
- **Host**: x86_64-unknown-linux-gnu
- **CPU**: AMD EPYC 7763 64-Core Processor

## Configuration

- **Capacity**: 4096
- **Universe**: 16384
- **Operations**: 200000
- **Seed**: 42

## Hit Rate Comparison

| Policy | Uniform | HotSet 90/10 | Scan | Zipfian 1.0 | Scrambled Zipfian | Latest | Scan Resistance | Flash Crowd |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| **2Q** | 24.78% | 90.63% | 0.00% | 82.37% | 90.71% | 31.55% | 16.04% | 89.36% |
| **Clock** | 24.66% | 90.65% | 0.00% | 80.75% | 90.45% | 34.96% | 17.63% | 88.74% |
| **Clock-Pro** | 24.77% | 90.65% | 0.00% | 81.54% | 90.38% | 30.55% | 21.09% | 89.14% |
| **FIFO** | 24.64% | 86.16% | 0.00% | 76.77% | 87.58% | 35.49% | 15.58% | 86.74% |
| **Heap-LFU** | 24.67% | 90.67% | 22.52% | 74.81% | 90.03% | 26.51% | 23.42% | 85.32% |
| **LFU** | 24.61% | 90.64% | 0.00% | 82.57% | 91.13% | 25.82% | 20.28% | 89.40% |
| **LIFO** | 24.76% | 90.66% | 24.22% | 80.25% | 90.18% | 26.08% | 29.10% | 63.08% |
| **LRU** | 24.62% | 90.65% | 0.00% | 80.17% | 90.22% | 35.45% | 16.75% | 88.50% |
| **LRU-K** | 24.61% | 90.64% | 0.00% | 82.57% | 91.13% | 25.91% | 20.28% | 89.40% |
| **MFU** | 24.66% | 37.18% | 18.44% | 31.99% | 59.85% | 24.89% | 23.42% | 41.69% |
| **MRU** | 24.74% | 38.07% | 24.22% | 33.10% | 59.53% | 24.90% | 28.90% | 39.71% |
| **NRU** | 24.74% | 90.66% | 24.21% | 80.29% | 90.20% | 26.35% | 29.10% | 88.17% |
| **Random** | 24.66% | 86.34% | 1.82% | 76.98% | 87.70% | 32.85% | 20.72% | 86.84% |
| **S3-FIFO** | 24.73% | 90.63% | 0.00% | 82.49% | 90.95% | 30.35% | 23.18% | 89.48% |
| **SLRU** | 24.75% | 90.63% | 0.00% | 82.57% | 90.89% | 30.95% | 16.92% | 89.40% |

## Throughput (Million ops/sec)

| Policy | Uniform | HotSet 90/10 | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 16.21 | 11.71 | 8.41 |
| **Clock** | 12.39 | 11.51 | 7.92 |
| **Clock-Pro** | 10.01 | 12.21 | 8.01 |
| **FIFO** | 11.68 | 11.30 | 7.94 |
| **Heap-LFU** | 7.51 | 7.98 | 6.02 |
| **LFU** | 8.24 | 7.54 | 5.46 |
| **LIFO** | 20.50 | 14.05 | 9.45 |
| **LRU** | 15.48 | 11.15 | 8.24 |
| **LRU-K** | 11.33 | 9.41 | 7.15 |
| **MFU** | 10.47 | 9.86 | 7.85 |
| **MRU** | 16.86 | 15.13 | 11.04 |
| **NRU** | 0.68 | 12.92 | 7.22 |
| **Random** | 16.36 | 13.74 | 9.08 |
| **S3-FIFO** | 11.17 | 11.25 | 7.79 |
| **SLRU** | 16.17 | 11.71 | 8.42 |

## Latency P99 (nanoseconds)

| Policy | Uniform | HotSet 90/10 | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 130 | 80 | 101 |
| **Clock** | 200 | 150 | 171 |
| **Clock-Pro** | 321 | 210 | 291 |
| **FIFO** | 210 | 161 | 180 |
| **Heap-LFU** | 351 | 261 | 311 |
| **LFU** | 380 | 191 | 220 |
| **LIFO** | 81 | 61 | 80 |
| **LRU** | 130 | 90 | 111 |
| **LRU-K** | 201 | 150 | 160 |
| **MFU** | 220 | 201 | 210 |
| **MRU** | 130 | 100 | 101 |
| **NRU** | 10420 | 160 | 521 |
| **Random** | 130 | 100 | 120 |
| **S3-FIFO** | 231 | 151 | 200 |
| **SLRU** | 140 | 80 | 100 |

## Scan Resistance

| Policy | Baseline | During Scan | Recovery | Score |
|--------|---------:|------------:|---------:|------:|
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

*Score = Recovery/Baseline (1.0 = perfect recovery, n/a = baseline too low to compare)*

## Adaptation Speed

| Policy | Stable Hit Rate | Ops to 50% | Ops to 80% |
|--------|----------------:|-----------:|-----------:|
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
| **Low latency required** | LRU, Clock | Fast operations, near O(1) average overhead |
| **Memory constrained** | LRU, Clock | Minimal metadata overhead |
| **Frequency-aware** | LFU, Heap-LFU, LRU-K | Track access frequency for better decisions |
| **Shifting patterns** | S3-FIFO, 2Q | Adapt to changing access patterns |
| **Multi-access patterns** | 2Q, S3-FIFO | Handle mixed one-hit and frequent items |

---

*Generated by `bench-support v0.1.0` from `results.json` (schema v1.0.0).*
