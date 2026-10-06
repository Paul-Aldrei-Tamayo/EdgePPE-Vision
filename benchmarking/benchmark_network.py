"""
EdgePPE-Vision: Network & REST API Benchmark Tool
Evaluates RESTful HTTP POST latency, Round Trip Time (RTT), and client FPS over LAN.
Matches targets in CPE179P Proposal Section 9.5 (RTT <= 250 ms, FPS >= 4-6).
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import requests

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def benchmark_api_network(server_url: str, image_path: str, count: int = 30):
    endpoint = f"{server_url.rstrip('/')}/api/v1/detect"
    print(f"\n[Network Benchmark] Server: {endpoint}")
    print(f"[Network Benchmark] Test Payload: {image_path}")
    print(f"[Network Benchmark] Running {count} requests...\n")

    if not Path(image_path).exists():
        print(f"Error: Sample image not found: {image_path}")
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    session = requests.Session()
    rtt_list = []
    inf_list = []
    net_list = []

    # Warmup
    try:
        session.post(endpoint, data=img_bytes, headers={"Content-Type": "image/jpeg"}, timeout=5.0)
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return

    for i in range(count):
        t0 = time.perf_counter()
        resp = session.post(endpoint, data=img_bytes, headers={"Content-Type": "image/jpeg"}, timeout=5.0)
        t1 = time.perf_counter()

        if resp.status_code == 200:
            data = resp.json()
            rtt = (t1 - t0) * 1000.0
            inf = data.get("inference_time_ms", 0.0)
            net = max(0.0, rtt - inf)

            rtt_list.append(rtt)
            inf_list.append(inf)
            net_list.append(net)
        else:
            print(f"Request {i+1} failed with status {resp.status_code}")

    if not rtt_list:
        print("No successful requests recorded.")
        return

    mean_rtt = np.mean(rtt_list)
    mean_inf = np.mean(inf_list)
    mean_net = np.mean(net_list)
    achieved_fps = 1000.0 / mean_rtt if mean_rtt > 0 else 0.0

    print("=" * 65)
    print("              NETWORK & API BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Requests Completed : {len(rtt_list)}")
    print(f"Mean Round Trip Time     : {mean_rtt:.2f} ms (Target: <= 250 ms)")
    print(f"Median Round Trip Time   : {np.median(rtt_list):.2f} ms")
    print(f"95th Percentile RTT      : {np.percentile(rtt_list, 95):.2f} ms")
    print(f"Mean Server Inference    : {mean_inf:.2f} ms")
    print(f"Mean Network Overhead    : {mean_net:.2f} ms")
    print(f"Estimated Max Frame Rate : {achieved_fps:.1f} FPS (Target: 4 - 6 FPS)")
    print("=" * 65)
    print(f"Status: {'PASS' if mean_rtt <= 250 else 'ATTENTION NEEDED'} (Target <= 250 ms RTT)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgePPE-Vision Network Benchmark")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=str, default=str(PROJECT_ROOT / "samples" / "images" / "building_workers_18.jpeg"))
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    benchmark_api_network(args.server, args.image, args.count)
