"""
EdgePPE-Vision: Raspberry Pi Edge Client Application
Course: CPE179P - AI on the Edge with Huawei Ascend and Atlas
Group 1: Go, Jonah Timothy; Tamayo, Paul Aldrei; Evangelista, Matthew

Captures camera frames (Picamera2 / USB Webcam / Video), sends HTTP POST requests
to the local Ubuntu Server VM FastAPI endpoint, and renders real-time bounding boxes,
compliance status banners, and system telemetry metrics.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import requests

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Check for Picamera2 availability on Raspberry Pi OS
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class EdgeCamera:
    """Unified camera abstraction supporting Picamera2, USB Webcam, and Video Stream."""
    def __init__(self, source: str = "0", width: int = 640, height: int = 480):
        self.source = source
        self.width = width
        self.height = height
        self.is_picam = False
        self.picam2 = None
        self.cap = None

        if source.lower() == "picam" or (source == "0" and PICAMERA2_AVAILABLE):
            try:
                print("[Camera] Initializing Raspberry Pi Camera Module via Picamera2...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                self.is_picam = True
                print("[Camera] Picamera2 initialized successfully.")
                return
            except Exception as e:
                print(f"[Camera] Picamera2 initialization failed: {e}. Falling back to OpenCV.")

        # Fallback to OpenCV VideoCapture (Webcam index or Video file path)
        cam_idx = int(source) if source.isdigit() else source
        print(f"[Camera] Initializing OpenCV VideoCapture from source: '{source}'...")
        self.cap = cv2.VideoCapture(cam_idx)
        if str(source).isdigit():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.is_picam and self.picam2:
            frame_rgb = self.picam2.capture_array()
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret and not str(self.source).isdigit():
                # Loop video files for continuous demonstration
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def release(self):
        if self.is_picam and self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap:
            self.cap.release()


class RaspberryPiPPEClient:
    """
    Client terminal running on Raspberry Pi 4/5.
    Connects to the FastAPI Inference Server over LAN.
    """
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        camera_source: str = "0",
        width: int = 640,
        height: int = 480,
        conf_threshold: float = 0.35,
        target_fps: float = 6.0,
        headless: bool = False
    ):
        self.server_url = server_url.rstrip("/")
        self.detect_endpoint = f"{self.server_url}/api/v1/detect"
        self.health_endpoint = f"{self.server_url}/api/v1/health"
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.conf_threshold = conf_threshold
        self.target_fps = target_fps
        self.headless = headless
        self.show_hud = True

        # Persistent HTTP session with pooling for low network latency
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Snapshot storage directory
        self.snapshot_dir = PROJECT_ROOT / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def check_server_connection(self) -> bool:
        """Verify reachability of the FastAPI inference node"""
        print(f"[Client] Testing connection to inference server: {self.health_endpoint}...")
        try:
            resp = self.session.get(self.health_endpoint, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[Client] Connected to Server! Model: {data.get('model_loaded')} | Quantization: {data.get('quantization')}")
                return True
            else:
                print(f"[Client] Warning: Server responded with status {resp.status_code}")
                return False
        except Exception as e:
            print(f"[Client] Connection failed: {e}")
            print(f"[Client] Ensure the Ubuntu VM FastAPI server is running on {self.server_url}")
            return False

    def send_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Compress frame to JPEG and send HTTP POST request to FastAPI server.
        Returns:
            (response_json, round_trip_time_ms)
        """
        # Compress frame to JPEG buffer
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return None, 0.0

        t0 = time.perf_counter()
        try:
            # Transmit raw binary JPEG bytes directly over LAN for highest throughput
            url = f"{self.detect_endpoint}?conf_threshold={self.conf_threshold}"
            resp = self.session.post(
                url,
                data=buffer.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=4.0
            )
            t1 = time.perf_counter()
            rtt_ms = (t1 - t0) * 1000.0

            if resp.status_code == 200:
                return resp.json(), round(rtt_ms, 1)
            else:
                print(f"[Client API Error] Server returned HTTP {resp.status_code}: {resp.text}")
                return None, round(rtt_ms, 1)
        except Exception as e:
            t1 = time.perf_counter()
            rtt_ms = (t1 - t0) * 1000.0
            return None, round(rtt_ms, 1)

    def render_overlay(
        self,
        frame: np.ndarray,
        api_data: Optional[Dict[str, Any]],
        rtt_ms: float,
        fps: float
    ) -> np.ndarray:
        """
        Draws bounding boxes, compliance alert banners, and telemetry metrics.
        Color coding: Green for compliant PPE, Red for violations.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        status = "CONNECTING" if api_data is None else api_data.get("status", "UNKNOWN")
        inference_ms = api_data.get("inference_time_ms", 0.0) if api_data else 0.0
        detections = api_data.get("detections", []) if api_data else []

        # 1. Draw Detections
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            name = det.get("class_name", "Object")
            conf = det.get("confidence", 0.0)

            x1, y1, x2, y2 = bbox
            is_violation = name.upper().startswith("NO-") or "violation" in name.lower()
            box_color = (0, 0, 255) if is_violation else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label badge
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(y1, th + 8)
            cv2.rectangle(annotated, (x1, y_label - th - 4), (x1 + tw + 4, y_label + 2), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y_label - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255) if is_violation else (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # 2. Draw Top Compliance Status Banner
        if status == "COMPLIANT":
            banner_color = (0, 180, 0)     # Green
            banner_text = "STATUS: COMPLIANT - ALL GEAR VERIFIED"
        elif status == "NON_COMPLIANT":
            banner_color = (0, 0, 220)     # Red
            banner_text = "STATUS: NON-COMPLIANT - VIOLATION DETECTED"
        else:
            banner_color = (0, 140, 255)   # Amber / Orange
            banner_text = f"STATUS: {status}"

        cv2.rectangle(annotated, (0, 0), (w, 34), banner_color, -1)
        cv2.putText(
            annotated,
            banner_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 3. Draw Bottom Telemetry HUD
        if self.show_hud:
            hud_bg_y = h - 28
            cv2.rectangle(annotated, (0, hud_bg_y), (w, h), (20, 24, 30), -1)
            net_overhead_ms = max(0.0, rtt_ms - inference_ms)
            hud_str = f"FPS: {fps:4.1f} | RTT: {rtt_ms:4.1f}ms | ONNX Inf: {inference_ms:4.1f}ms | Net: {net_overhead_ms:4.1f}ms | Conf: {self.conf_threshold}"
            cv2.putText(
                annotated,
                hud_str,
                (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 220, 240),
                1,
                cv2.LINE_AA
            )

        return annotated

    def run(self):
        """Main execution loop for Raspberry Pi"""
        print("\n" + "=" * 65)
        print("  EdgePPE-Vision Raspberry Pi Edge Client")
        print(f"  Target Server : {self.server_url}")
        print(f"  Video Source  : {self.camera_source}")
        print(f"  Resolution    : {self.width}x{self.height}")
        print(f"  Target FPS    : {self.target_fps}")
        print("=" * 65)

        self.check_server_connection()

        try:
            cam = EdgeCamera(self.camera_source, self.width, self.height)
        except Exception as e:
            print(f"[Client Fatal] Could not open camera: {e}")
            return

        window_name = "EdgePPE-Vision | Raspberry Pi Edge Terminal"
        if not self.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.width, self.height)

        prev_time = time.perf_counter()
        fps_smoothing = 0.0

        print("\n[Client Running] Controls: 'q'=quit, 's'=snapshot, 'c'=cycle conf, 'h'=toggle HUD\n")

        try:
            while True:
                loop_start = time.perf_counter()
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                # Ensure frame matches desired dimensions
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Send frame to FastAPI inference server
                api_data, rtt_ms = self.send_frame(frame)

                # Compute client frame rate
                loop_end = time.perf_counter()
                frame_time = loop_end - prev_time
                prev_time = loop_end
                instant_fps = 1.0 / max(frame_time, 1e-5)
                fps_smoothing = (0.8 * fps_smoothing) + (0.2 * instant_fps) if fps_smoothing > 0 else instant_fps

                # Render overlays
                display_frame = self.render_overlay(frame, api_data, rtt_ms, fps_smoothing)

                if self.headless:
                    status = api_data.get("status") if api_data else "ERR"
                    dets = len(api_data.get("detections", [])) if api_data else 0
                    print(f"[Telemetry] Status: {status:14s} | Dets: {dets} | RTT: {rtt_ms:.1f}ms | FPS: {fps_smoothing:.1f}")
                else:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[Client] Quitting...")
                        break
                    elif key == ord("s"):
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        snap_file = self.snapshot_dir / f"snapshot_{ts}.jpg"
                        cv2.imwrite(str(snap_file), display_frame)
                        print(f"[Client] Saved snapshot to: {snap_file}")
                    elif key == ord("c"):
                        thresholds = [0.25, 0.35, 0.45, 0.55]
                        curr_idx = thresholds.index(self.conf_threshold) if self.conf_threshold in thresholds else 0
                        self.conf_threshold = thresholds[(curr_idx + 1) % len(thresholds)]
                        print(f"[Client] Updated confidence threshold to: {self.conf_threshold}")
                    elif key == ord("h"):
                        self.show_hud = not self.show_hud

                # Frame pacing to maintain target FPS without overloading edge CPU
                target_frame_time = 1.0 / self.target_fps
                elapsed = time.perf_counter() - loop_start
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        except KeyboardInterrupt:
            print("\n[Client] Interrupted by user.")
        finally:
            cam.release()
            if not self.headless:
                cv2.destroyAllWindows()
            print("[Client] Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="EdgePPE-Vision Raspberry Pi Edge Client")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="FastAPI Server base URL")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for USB/Pi camera, or path to .mp4)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--fps", type=float, default=6.0, help="Target FPS (default 6 FPS)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    args = parser.parse_args()

    client = RaspberryPiPPEClient(
        server_url=args.server,
        camera_source=args.source,
        width=args.width,
        height=args.height,
        conf_threshold=args.conf,
        target_fps=args.fps,
        headless=args.headless
    )
    client.run()


if __name__ == "__main__":
    main()
