"""
EdgePPE-Vision: FastAPI Inference Microservice
Course: CPE179P - AI on the Edge with Huawei Ascend and Atlas
Group 1: Go, Jonah Timothy; Tamayo, Paul Aldrei; Evangelista, Matthew

Exposes high-performance asynchronous REST endpoints for Raspberry Pi edge clients.
Provides ONNX Runtime acceleration, INT8 quantization support, and real-time compliance evaluation.
"""

import os
import io
import time
import base64
import psutil
from typing import Optional, List
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Query, HTTPException, status
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (
    APP_DIR,
    BACKEND_DIR,
    PROJECT_ROOT,
    FP32_MODEL_PATH,
    INT8_MODEL_PATH,
    ACTIVE_MODEL,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MANDATORY_GEAR,
    VIOLATION_CLASSES
)
from .schemas import DetectionResponse, ServerHealthResponse, DetectionItem, ComplianceDetails
from .detector import ONNXDetector
from .compliance import ComplianceEngine

# Initialize FastAPI App
app = FastAPI(
    title="EdgePPE-Vision Inference API",
    description="A Quantized Edge AI System for Real-Time PPE Compliance Monitoring via FastAPI and ONNX Runtime",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local edge network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates & Static Files
templates_dir = BACKEND_DIR / "templates"
static_dir = BACKEND_DIR / "static"
samples_dir = PROJECT_ROOT / "samples"

if not templates_dir.exists():
    templates_dir.mkdir(parents=True, exist_ok=True)
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
if samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")

# Global Inference & Compliance Engines
use_int8 = (ACTIVE_MODEL.lower() == "int8") and os.path.exists(INT8_MODEL_PATH)
initial_model = INT8_MODEL_PATH if use_int8 else FP32_MODEL_PATH

try:
    detector = ONNXDetector(model_path=initial_model)
    print(f"[FastAPI Server] Model loaded successfully: {initial_model} (INT8: {detector.is_quantized})")
except Exception as e:
    print(f"[FastAPI Server] Error initializing detector: {e}")
    detector = None

compliance_engine = ComplianceEngine(
    mandatory_gear=MANDATORY_GEAR,
    violation_classes=VIOLATION_CLASSES
)


async def decode_image_request(request: Request, file: Optional[UploadFile] = None) -> np.ndarray:
    """Helper to decode image from multipart file or raw POST bytes"""
    image_bytes = None
    if file is not None:
        image_bytes = await file.read()
    else:
        # Check raw request body (used for high-speed edge streaming from Raspberry Pi)
        image_bytes = await request.body()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image data received. Provide a multipart file or raw JPEG body."
        )

    # Decode image from buffer
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to decode image bytes into a valid visual frame."
        )
    return image


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Interactive EdgePPE-Vision Web Dashboard for live testing & demonstration"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_type": "INT8 Quantized" if (detector and detector.is_quantized) else "FP32 Baseline",
            "classes": detector.labels if detector else [],
            "mandatory": MANDATORY_GEAR,
            "violations": VIOLATION_CLASSES
        }
    )


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_ppe(
    request: Request,
    file: Optional[UploadFile] = File(None),
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    iou_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    enforce_mask: bool = Query(False, description="Flag NO-Mask as a safety violation if True")
):
    """
    Primary REST endpoint for Raspberry Pi client.
    Accepts JPEG frames over LAN, executes ONNX Runtime inference, evaluates compliance,
    and returns structured JSON detection coordinates and compliance status.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

    image = await decode_image_request(request, file)

    # Run ONNX inference
    detections, latency_ms = detector.detect(
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    # Evaluate compliance
    compliance_result = compliance_engine.evaluate(detections, enforce_mask=enforce_mask)

    return DetectionResponse(
        status=compliance_result["status"],
        inference_time_ms=latency_ms,
        total_detected=len(detections),
        detections=[DetectionItem(**d) for d in detections],
        compliance_details=ComplianceDetails(
            is_compliant=compliance_result["is_compliant"],
            mandatory_present=compliance_result["mandatory_present"],
            missing_mandatory=compliance_result["missing_mandatory"],
            violations_detected=compliance_result["violations_detected"],
            summary=compliance_result["summary"]
        )
    )


@app.post("/api/v1/detect_visual")
async def detect_visual(
    request: Request,
    file: Optional[UploadFile] = File(None),
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    iou_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    enforce_mask: bool = Query(False, description="Flag NO-Mask as a safety violation if True"),
    return_base64: bool = Query(False)
):
    """
    Visual detection endpoint: Returns the annotated frame with bounding boxes and compliance HUD.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

    image = await decode_image_request(request, file)
    detections, latency_ms = detector.detect(image, conf_threshold, iou_threshold)
    compliance_result = compliance_engine.evaluate(detections, enforce_mask=enforce_mask)

    annotated = detector.draw_overlays(image, detections, status=compliance_result["status"])

    # Add HUD latency text
    hud_text = f"Latency: {latency_ms:.1f}ms | Mode: {'INT8' if detector.is_quantized else 'FP32'}"
    cv2.putText(annotated, hud_text, (10, annotated.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated frame.")

    if return_base64:
        encoded = base64.b64encode(buffer).decode("utf-8")
        return {
            "status": compliance_result["status"],
            "inference_time_ms": latency_ms,
            "total_detected": len(detections),
            "image_base64": f"data:image/jpeg;base64,{encoded}",
            "compliance_details": compliance_result
        }

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/api/v1/health", response_model=ServerHealthResponse)
@app.get("/health")
async def health_check():
    """
    Server health and edge node telemetry endpoint.
    Reports RAM usage, CPU load, and active model quantization mode.
    """
    mem = psutil.virtual_memory()
    return ServerHealthResponse(
        status="healthy" if detector else "degraded",
        model_loaded=os.path.basename(detector.model_path) if detector else "none",
        quantization="INT8 Quantized" if (detector and detector.is_quantized) else "FP32 Baseline",
        device="CPU (ONNX Runtime CPUExecutionProvider)",
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_used_mb=round(mem.used / (1024 * 1024), 2),
        ram_total_mb=round(mem.total / (1024 * 1024), 2)
    )


@app.get("/api/v1/info")
async def get_system_info():
    """Returns detailed EdgePPE-Vision model configuration and classes"""
    return {
        "project": "EdgePPE-Vision: A Quantized Edge AI System for Real-Time PPE Compliance Monitoring",
        "course": "CPE179P - AI on the Edge with Huawei Ascend and Atlas",
        "active_model": os.path.basename(detector.model_path) if detector else None,
        "quantization": "INT8" if (detector and detector.is_quantized) else "FP32",
        "input_resolution": f"{detector.img_size}x{detector.img_size}" if detector else "640x640",
        "target_classes": detector.labels if detector else [],
        "mandatory_gear": MANDATORY_GEAR,
        "violation_classes": VIOLATION_CLASSES,
        "threads": detector.num_threads if detector else 1
    }


@app.post("/api/v1/switch_model")
async def switch_model(use_int8: bool = Query(..., description="True for INT8, False for FP32")):
    """Dynamically switch between FP32 baseline and INT8 quantized model for benchmarking"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not ready.")

    success = detector.switch_model(use_int8=use_int8)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Target model file not found ({'INT8' if use_int8 else 'FP32'}). Check models/ directory."
        )

    return {
        "success": True,
        "active_model": os.path.basename(detector.model_path),
        "is_quantized": detector.is_quantized
    }
