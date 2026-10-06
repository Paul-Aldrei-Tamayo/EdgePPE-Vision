"""
EdgePPE-Vision: Pydantic Data Models & Schemas
Matches the API contracts specified in CPE179P Proposal Page 11 & 14.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    class_name: str = Field(..., description="Detected PPE or violation category name")
    confidence: float = Field(..., description="Detection confidence score between 0.0 and 1.0")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2] in original image space")


class ComplianceDetails(BaseModel):
    is_compliant: bool = Field(..., description="True if mandatory gear is worn and no violations detected")
    mandatory_present: List[str] = Field(default_factory=list, description="Mandatory PPE items detected")
    missing_mandatory: List[str] = Field(default_factory=list, description="Mandatory PPE items missing")
    violations_detected: List[str] = Field(default_factory=list, description="Violation classes detected (NO-*)")
    summary: str = Field(..., description="Human-readable safety compliance summary")


class DetectionResponse(BaseModel):
    status: str = Field(..., description="'COMPLIANT' or 'NON_COMPLIANT'")
    inference_time_ms: float = Field(..., description="ONNX model execution time in milliseconds")
    total_detected: int = Field(..., description="Total number of detected objects")
    detections: List[DetectionItem] = Field(default_factory=list, description="List of detected items")
    compliance_details: Optional[ComplianceDetails] = Field(None, description="Detailed breakdown of compliance")


class ServerHealthResponse(BaseModel):
    status: str
    service: str = "EdgePPE-Vision Inference Server"
    model_loaded: str
    quantization: str
    device: str
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
