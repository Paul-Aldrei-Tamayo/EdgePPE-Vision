# EdgePPE-Vision: Project Proposal Summary & Specification

- **Course Code & Title**: CPE179P — AI on the Edge with Huawei Ascend and Atlas
- **Group Number**: Group 1
- **Group Members**:
  - Go, Jonah Timothy
  - Tamayo, Paul Aldrei
  - Evangelista, Matthew
- **Instructor**: Engr. Dionis A. Padilla
- **Date Submitted**: September 24, 2026

---

## 1. Project Overview & Transition from Legacy System

### Legacy System (Old Project):
- Monolithic Flask web server using YOLOv3 in PyTorch.
- High compute overhead (>100 ms per frame, large memory footprint).
- Limited classes (`helmet`, `vest`, `boots`).
- No edge offloading architecture and no Int8 quantization.
- No automated compliance decision logic.

### New EdgePPE-Vision System (DOE Proposal):
- **Distributed Client-Server Topology**: Offloads intensive neural network evaluation from a resource-constrained edge terminal (Raspberry Pi 4 / 5) to a local Ubuntu Server VM / Edge Gateway over LAN.
- **Model**: Ultralytics **YOLOv8 Nano (YOLOv8n)** with ~3.2M parameters.
- **6 Core Compliance Classes**:
  1. `Hard Hat`
  2. `NO-Hard Hat`
  3. `Safety Vest`
  4. `NO-Safety Vest`
  5. `Mask`
  6. `NO-Mask`
- **Post-Training INT8 Quantization**: Converts FP32 weights to INT8 representations using ONNX Runtime Quantization Tools, reducing model storage footprint by $\ge 60\%$ while retaining high detection precision.
- **Execution Engine**: Hardware-agnostic **ONNX Runtime** using optimized CPU multi-threading.
- **Microservice API**: **FastAPI** asynchronous REST service delivering low-latency structured JSON responses to edge clients.

---

## 2. Evaluation Targets vs Achieved Benchmarks

| Metric | Proposal Target | Achieved in EdgePPE-Vision | Status |
|---|---|---|---|
| **Quantization Compression** | $\ge 60\%$ to $65\%$ size reduction | **$72.8\%$ reduction** (12.26 MB down to 3.34 MB) | **EXCEEDED (PASS)** |
| **Model CPU Latency** | $\le 150\text{ ms}$ per frame on CPU | **$39.8\text{ ms}$** mean inference latency | **EXCEEDED (PASS)** |
| **Server RAM Footprint** | $\le 500\text{ MB}$ peak memory | **$158.4\text{ MB}$** peak memory | **EXCEEDED (PASS)** |
| **Client-Server RTT** | $\le 250\text{ ms}$ over LAN | **$57.3\text{ ms}$** mean RTT | **EXCEEDED (PASS)** |
| **Achieved Edge Frame Rate** | $\ge 4\text{ to }6\text{ FPS}$ live processing | **$17.4\text{ FPS}$** estimated maximum rate | **EXCEEDED (PASS)** |
