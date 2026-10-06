# EdgePPE-Vision Performance Benchmark Report

- **Course**: CPE179P - AI on the Edge with Huawei Ascend and Atlas
- **Group**: Group 1 (Go, Tamayo, Evangelista)
- **Test Platform**: CPU Execution Provider (24 threads)

## 1. Storage & Compression Performance

| Metric | FP32 Baseline | INT8 Quantized | Proposal Target | Status |
|---|---|---|---|---|
| **Model Size** | `11.7 MB` | `3.2 MB` | `>= 60% reduction` | **72.6% reduction (PASS)** |
| **Peak RAM** | `161.4 MB` | `156.9 MB` | `<= 500 MB` | **PASS** |

## 2. Inference Latency Performance

| Metric | FP32 Baseline | INT8 Quantized |
|---|---|---|
| **Mean Latency** | `14.95 ms` | `32.41 ms` |
| **Median Latency** | `14.69 ms` | `31.77 ms` |
| **Min Latency** | `12.79 ms` | `30.26 ms` |
| **Max Latency** | `20.35 ms` | `41.06 ms` |
| **95th Percentile** | `20.1 ms` | `37.49 ms` |
