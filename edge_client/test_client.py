"""
EdgePPE-Vision: CLI Test Client
Sends a single test image or folder of images to the FastAPI server and prints JSON response.

Usage:
    python test_client.py --image ../samples/images/building_workers_18.jpeg --server http://127.0.0.1:8000
"""

import sys
import argparse
import json
from pathlib import Path
import requests

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def test_image(image_path: str, server_url: str = "http://127.0.0.1:8000", conf: float = 0.35):
    endpoint = f"{server_url.rstrip('/')}/api/v1/detect?conf_threshold={conf}"
    print(f"\n[TestClient] Testing image: {image_path}")
    print(f"[TestClient] Endpoint: {endpoint}")

    if not Path(image_path).exists():
        print(f"Error: Image file does not exist: {image_path}")
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    resp = requests.post(
        endpoint,
        data=img_bytes,
        headers={"Content-Type": "image/jpeg"},
        timeout=5.0
    )

    print(f"[TestClient] HTTP Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print("\n--- Detection Response ---")
        print(f"Compliance Status : {data.get('status')}")
        print(f"Inference Latency : {data.get('inference_time_ms')} ms")
        print(f"Total Detections  : {data.get('total_detected')}")
        print("Detections List:")
        for det in data.get("detections", []):
            print(f"  - {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
        print(f"Compliance Summary: {data.get('compliance_details', {}).get('summary')}")
        print("--------------------------\n")
    else:
        print(f"Error Response: {resp.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EdgePPE-Vision REST API")
    parser.add_argument("--image", type=str, default=str(PROJECT_ROOT / "samples" / "images" / "building_workers_18.jpeg"))
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    test_image(args.image, args.server, args.conf)
