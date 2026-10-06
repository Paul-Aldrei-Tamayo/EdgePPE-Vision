# EdgePPE-Vision REST API Reference

The **EdgePPE-Vision** backend exposes high-performance asynchronous REST endpoints powered by FastAPI and ONNX Runtime.

Base URL: `http://<server-ip>:8000`  
Interactive Swagger Documentation: `http://<server-ip>:8000/docs`

---

## 1. Primary Inference Endpoint

### `POST /api/v1/detect`
Performs object detection using the quantized INT8 YOLOv8n ONNX model, evaluates workplace PPE compliance rules, and returns bounding coordinates with compliance state.

#### Query Parameters:
- `conf_threshold` *(float, optional)*: Minimum confidence threshold (0.0 to 1.0, default: `0.35`).
- `iou_threshold` *(float, optional)*: Intersection-over-Union NMS threshold (default: `0.45`).

#### Request Formats Supported:
1. **Raw Binary JPEG Buffer** *(Fastest for Raspberry Pi stream)*:
   - Header: `Content-Type: image/jpeg`
   - Body: Raw binary JPEG bytes.
2. **Multipart Form-Data** *(Standard web file upload)*:
   - Form field name: `file`

#### Response: `200 OK` (application/json)
```json
{
  "status": "NON_COMPLIANT",
  "inference_time_ms": 42.7,
  "total_detected": 2,
  "detections": [
    {
      "class_name": "Hard Hat",
      "confidence": 0.91,
      "bbox": [232, 126, 439, 483]
    },
    {
      "class_name": "NO-Safety Vest",
      "confidence": 0.84,
      "bbox": [210, 320, 480, 610]
    }
  ],
  "compliance_details": {
    "is_compliant": false,
    "mandatory_present": [
      "Hard Hat"
    ],
    "missing_mandatory": [
      "Safety Vest"
    ],
    "violations_detected": [
      "NO-Safety Vest"
    ],
    "summary": "Safety Violation: Detected NO-Safety Vest."
  }
}
```

---

## 2. Visual Annotated Endpoint

### `POST /api/v1/detect_visual`
Accepts an image and returns the annotated visual frame with color-coded bounding boxes and compliance HUD.

#### Query Parameters:
- `conf_threshold` *(float, optional)*: Confidence threshold.
- `return_base64` *(bool, optional, default: false)*: If true, returns JSON containing `image_base64`. If false, returns raw JPEG binary (`image/jpeg`).

---

## 3. Health & Telemetry Endpoint

### `GET /api/v1/health` or `GET /health`
Returns node telemetry including RAM usage, CPU load, and model status.

#### Response: `200 OK`
```json
{
  "status": "healthy",
  "service": "EdgePPE-Vision Inference Server",
  "model_loaded": "yolov8n_ppe_int8.onnx",
  "quantization": "INT8 Quantized",
  "device": "CPU (ONNX Runtime CPUExecutionProvider)",
  "cpu_percent": 18.4,
  "ram_used_mb": 158.4,
  "ram_total_mb": 4096.0
}
```

---

## 4. System Info Endpoint

### `GET /api/v1/info`
Returns class definitions, active model parameters, and compliance rules.

#### Response: `200 OK`
```json
{
  "project": "EdgePPE-Vision: A Quantized Edge AI System for Real-Time PPE Compliance Monitoring",
  "course": "CPE179P - AI on the Edge with Huawei Ascend and Atlas",
  "active_model": "yolov8n_ppe_int8.onnx",
  "quantization": "INT8",
  "input_resolution": "640x640",
  "target_classes": [
    "Hard Hat",
    "NO-Hard Hat",
    "Safety Vest",
    "NO-Safety Vest",
    "Mask",
    "NO-Mask"
  ],
  "mandatory_gear": [
    "Hard Hat",
    "Safety Vest"
  ],
  "violation_classes": [
    "NO-Hard Hat",
    "NO-Safety Vest",
    "NO-Mask"
  ]
}
```

---

## 5. Model Switcher Endpoint (Benchmarking)

### `POST /api/v1/switch_model?use_int8=true`
Switches live between FP32 baseline and INT8 quantized ONNX models without restarting the service.
