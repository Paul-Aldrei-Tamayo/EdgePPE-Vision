# EdgePPE-Vision: A Quantized Edge AI System for Real-Time PPE Compliance Monitoring via FastAPI and ONNX Runtime

**Course**: CPE179P — AI on the Edge with Huawei Ascend and Atlas  
**Group Number**: Group 1  
**Group Members**:
- Go, Jonah Timothy
- Tamayo, Paul Aldrei
- Evangelista, Matthew  

**Instructor**: Engr. Dionis A. Padilla  
**Date Submitted**: September 24, 2026  

---

## 📌 Executive Summary

In industrial facilities, construction zones, and manufacturing sites, manual checking of Personal Protective Equipment (PPE) compliance is inefficient and subject to human fatigue. **EdgePPE-Vision** is a lightweight, localized client-server Edge AI system engineered to perform automated real-time compliance monitoring of mandatory safety gear (**Hard Hat**, **Safety Vest**, **Mask**) and flag violations (**NO-Hard Hat**, **NO-Safety Vest**, **NO-Mask**) without relying on external cloud connectivity.

The system utilizes an optimized **YOLOv8 Nano (YOLOv8n)** model converted to **ONNX** format and compressed via **Post-Training INT8 Quantization** ($\ge 70\%$ storage reduction). The backend microservice is hosted on an **Ubuntu Server Virtual Machine** using **FastAPI** and **ONNX Runtime** for CPU-accelerated inference. A **Raspberry Pi 4 / 5** equipped with a Camera Module serves as the edge client: capturing video frames, sending RESTful HTTP POST requests over local LAN, and rendering real-time bounding box overlays and compliance HUD alerts.

---

## 🏗️ System Architecture

```text
+-------------------------------------------------------------------------+
|                       EDGE DEVICE CLIENT                                |
|                    (Raspberry Pi 4 / Pi 5)                              |
|                                                                         |
|  [ Pi Camera / USB Webcam ]                [ Display Monitor / HUD ]    |
|      Frame Capture                               BBox & Compliance View |
+-------------------------------------------------------------------------+
                    |                                        ^
                    | POST /api/v1/detect                    | HTTP 200 OK
                    | (JPEG Buffer Bytes)                    | (JSON Predictions)
                    v                                        |
+-------------------------------------------------------------------------+
|                         COMMUNICATION LAYER                             |
|                     RESTful API over HTTP / JSON                        |
+-------------------------------------------------------------------------+
                    |                                        ^
                    v                                        |
+-------------------------------------------------------------------------+
|                        INFERENCE NODE SERVER                            |
|             (Ubuntu 22.04 LTS VM / VirtualBox / PC)                     |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                        FastAPI Async Service                      |  |
|  |  Image Ingestion & Tensor Preprocess  |  Compliance Logic Engine  |  |
|  +-------------------------------------------------------------------+  |
|                                   |                                     |
|                                   v                                     |
|  +-------------------------------------------------------------------+  |
|  |                    ONNX Runtime Inference Engine                  |  |
|  |             YOLOv8n Safety Gear Detector (INT8 Quantized)         |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

---

## 🎯 Target Classes & Compliance Decision Logic

### 6 Core Categories:
1. `Hard Hat` (Compliant Gear - Green Box)
2. `NO-Hard Hat` (Violation - Red Box)
3. `Safety Vest` (Compliant Gear - Green Box)
4. `NO-Safety Vest` (Violation - Red Box)
5. `Mask` (Compliant Gear - Cyan/Green Box)
6. `NO-Mask` (Violation - Red Box)

### Decision Logic Rules:
- **`NON_COMPLIANT`**: If **ANY** violation tag is detected (`NO-Hard Hat`, `NO-Safety Vest`, `NO-Mask`), or if mandatory equipment is missing when personnel are detected.
- **`COMPLIANT`**: If mandatory gear (`Hard Hat` AND `Safety Vest`) is detected with zero safety violations present.

---

## 📊 Benchmarking & Performance Metrics

| Metric | Proposal Target | Achieved in EdgePPE-Vision | Status |
|---|---|---|---|
| **Quantization Compression** | $\ge 60\%$ to $65\%$ size reduction | **$72.8\%$ reduction** (12.26 MB down to 3.34 MB) | **EXCEEDED (PASS)** |
| **Model Inference Latency** | $\le 150\text{ ms}$ on CPU | **$39.8\text{ ms}$** mean CPU latency | **EXCEEDED (PASS)** |
| **Server RAM Footprint** | $\le 500\text{ MB}$ peak memory | **$158.4\text{ MB}$** peak memory | **EXCEEDED (PASS)** |
| **API Round-Trip Time (RTT)** | $\le 250\text{ ms}$ over LAN | **$57.3\text{ ms}$** mean RTT | **EXCEEDED (PASS)** |
| **Edge Frame Rate (FPS)** | $\ge 4\text{ to }6\text{ FPS}$ live feed | **$17.4\text{ FPS}$** estimated maximum rate | **EXCEEDED (PASS)** |

---

## 📁 Repository Directory Structure

```text
PyTorch-Based-PPE-Detection-Go-Jonah-Timothy/
├── backend_server/                 # FastAPI & ONNX Runtime Microservice (Ubuntu VM / Host)
│   ├── app/
│   │   ├── main.py                # REST API endpoints, CORS, file ingestion, visual preview
│   │   ├── detector.py            # ONNX Runtime inference engine, letterbox, NMS
│   │   ├── compliance.py          # Workplace PPE compliance decision logic
│   │   ├── config.py              # Server settings, paths, thresholds, and CPU thread pool
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── templates/
│   │   └── index.html             # Interactive HTML5 Web Dashboard (live testing, webcam, telemetry)
│   ├── run_server.py              # Backend launcher script
│   └── requirements.txt           # Backend Python dependencies
├── edge_client/                    # Raspberry Pi 4 / 5 Edge Client
│   ├── pi_client.py               # Main edge terminal (Picamera2 / OpenCV, live HUD overlay)
│   ├── test_client.py             # Fast CLI tester for single images & REST verification
│   ├── requirements_pi.txt        # Lightweight client dependencies
│   ├── run_client.sh              # Raspberry Pi OS bash launcher
│   └── run_client.bat             # Windows testing batch launcher
├── models/                         # Pre-packaged ONNX models
│   ├── yolov8n_ppe.onnx           # FP32 ONNX Baseline Model (12.3 MB)
│   ├── yolov8n_ppe_int8.onnx      # Quantized INT8 ONNX Model (3.34 MB)
│   └── labels.json                # Class taxonomy and compliance mapping
├── ml_pipeline/                    # ML Pipeline, Training & Quantization
│   ├── export_and_quantize.py     # YOLOv8n to ONNX FP32 & INT8 conversion pipeline
│   ├── calibrate.py               # CalibrationDataReader for static INT8 quantization
│   ├── train_ppe.py               # YOLOv8n fine-tuning script on PPE datasets
│   └── data_ppe.yaml              # Roboflow / Pictor PPE dataset configuration
├── benchmarking/                   # Benchmarking & Performance Suite
│   ├── benchmark_models.py        # Compares FP32 vs INT8 size, CPU latency, RAM footprint
│   ├── benchmark_network.py       # Measures API Round Trip Time (RTT), overhead, and FPS
│   └── benchmark_report.md        # Automatically generated performance report
├── samples/                        # Test media
│   ├── images/                    # Sample PPE and violation images
│   └── videos/                    # Sample MP4 demonstration clips
├── docs/                           # Comprehensive Setup Guides & Specifications
│   ├── RASPBERRY_PI_SETUP.md      # Raspberry Pi 4/5 hardware & Picamera2 guide
│   ├── UBUNTU_VM_SETUP.md         # VirtualBox Ubuntu 22.04 LTS Bridged Network guide
│   ├── API_REFERENCE.md           # Full REST API contracts & sample JSON
│   └── EDGE_PPE_VISION_PROPOSAL.md# Project proposal specifications
├── PPE_detection_YOLO/            # Preserved legacy Flask + YOLOv3 application
└── README.md                      # Primary project documentation
```

---

## 🚀 Quick Start Guide

### 1. Setting Up the Backend Inference Server (Ubuntu VM / Host PC)

1. Open a terminal in `backend_server/`:
   ```bash
   cd backend_server
   pip install -r requirements.txt
   ```
2. Start the FastAPI microservice:
   ```bash
   python run_server.py --host 0.0.0.0 --port 8000 --model int8
   ```
3. Open your browser to access:
   - **Interactive Web Dashboard**: `http://localhost:8000/`
   - **Interactive Swagger Documentation**: `http://localhost:8000/docs`
   - **Server Health Telemetry**: `http://localhost:8000/api/v1/health`

---

### 2. Setting Up the Edge Client (Raspberry Pi 4 / 5)

1. Connect your Raspberry Pi to the same Wi-Fi / LAN network as the Ubuntu VM.
2. In the `edge_client/` folder on the Raspberry Pi:
   ```bash
   cd edge_client
   pip install -r requirements_pi.txt
   ```
3. Launch live camera monitoring:
   ```bash
   # Using Raspberry Pi Camera Module or USB Webcam:
   python pi_client.py --server http://<UBUNTU_VM_IP>:8000 --source 0 --fps 6.0

   # Or run a demonstration with pre-recorded video:
   python pi_client.py --server http://<UBUNTU_VM_IP>:8000 --source ../samples/videos/ppe-1.mp4
   ```

#### Client Controls:
- **`q`** : Quit the client application.
- **`s`** : Save timestamped snapshot to `snapshots/`.
- **`c`** : Toggle detection confidence threshold.
- **`h`** : Toggle telemetry HUD overlay.

---

### 3. Running Benchmarks

#### Model Compression & CPU Latency Benchmark:
```bash
python benchmarking/benchmark_models.py --iterations 30
```

#### Network RTT & Throughput Benchmark:
```bash
python benchmarking/benchmark_network.py --server http://localhost:8000 --count 30
```

---

## 🛠️ Re-Exporting or Quantizing Models

To re-export a fine-tuned YOLOv8 PyTorch model (`.pt`) to ONNX FP32 and generate the quantized INT8 model:
```bash
python ml_pipeline/export_and_quantize.py --weights yolov8n.pt --mode dynamic
```

---

## 📚 Documentation Links
- [Raspberry Pi 4/5 Client Setup Guide](file:///docs/RASPBERRY_PI_SETUP.md)
- [Ubuntu 22.04 LTS Server VM Setup Guide](file:///docs/UBUNTU_VM_SETUP.md)
- [REST API Reference](file:///docs/API_REFERENCE.md)
- [Proposal Specification Summary](file:///docs/EDGE_PPE_VISION_PROPOSAL.md)
