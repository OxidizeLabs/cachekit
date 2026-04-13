# Benchmark Results

**Quick Links**: [📊 Interactive Charts](charts.html) | [📁 Raw JSON](results.json)

---

## Environment

- **Date**: 2026-04-13T01:56:21.457952076+00:00
- **Commit**: `ff682713024d49c46963efba2ec9fc33871a9e62`
- **Branch**: `main`
- **Dirty**: false
- **Rustc**: rustc 1.94.1 (e408947bf 2026-03-25)
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
| **MFU** | 41.69% | 37.18% | 24.89% | 18.44% | 23.42% | 59.85% | 24.66% | 31.99% |
| **MRU** | 39.71% | 38.07% | 24.90% | 24.22% | 28.90% | 59.53% | 24.74% | 33.10% |
| **NRU** | 88.17% | 90.66% | 26.35% | 24.21% | 29.10% | 90.20% | 24.74% | 80.29% |
| **Random** | 86.84% | 86.34% | 32.85% | 1.82% | 20.72% | 87.70% | 24.66% | 76.98% |
| **S3-FIFO** | 89.48% | 90.63% | 30.35% | 0.00% | 23.18% | 90.95% | 24.73% | 82.49% |
| **SLRU** | 89.40% | 90.63% | 30.95% | 0.00% | 16.92% | 90.89% | 24.75% | 82.57% |

## Throughput (Million ops/sec)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 11.66 | 16.15 | 8.40 |
| **Clock** | 13.28 | 17.34 | 8.98 |
| **Clock-Pro** | 12.01 | 10.00 | 7.98 |
| **FIFO** | 11.19 | 11.65 | 7.81 |
| **Heap-LFU** | 7.84 | 7.56 | 5.95 |
| **LFU** | 7.33 | 8.14 | 5.35 |
| **LIFO** | 14.08 | 20.59 | 9.53 |
| **LRU** | 10.97 | 15.27 | 8.14 |
| **LRU-K** | 10.53 | 12.64 | 7.75 |
| **MFU** | 9.60 | 10.24 | 7.73 |
| **MRU** | 15.29 | 17.06 | 11.18 |
| **NRU** | 12.52 | 0.68 | 7.05 |
| **Random** | 13.71 | 16.22 | 9.16 |
| **S3-FIFO** | 10.96 | 11.12 | 7.51 |
| **SLRU** | 11.56 | 15.93 | 8.39 |

## Latency P99 (nanoseconds)

| Policy | HotSet 90/10 | Uniform | Zipfian 1.0 |
|--------|-------:|-------:|-------:|
| **2Q** | 90 | 140 | 101 |
| **Clock** | 81 | 130 | 110 |
| **Clock-Pro** | 240 | 320 | 281 |
| **FIFO** | 160 | 210 | 200 |
| **Heap-LFU** | 270 | 331 | 331 |
| **LFU** | 200 | 361 | 221 |
| **LIFO** | 61 | 81 | 80 |
| **LRU** | 100 | 131 | 120 |
| **LRU-K** | 140 | 190 | 151 |
| **MFU** | 250 | 221 | 241 |
| **MRU** | 101 | 140 | 110 |
| **NRU** | 170 | 10349 | 571 |
| **Random** | 100 | 130 | 111 |
| **S3-FIFO** | 230 | 250 | 281 |
| **SLRU** | 90 | 141 | 110 |

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
