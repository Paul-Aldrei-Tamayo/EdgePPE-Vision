"""
EdgePPE-Vision: Backend Server Launcher
Launches the FastAPI REST inference microservice with Uvicorn.

Usage:
    python run_server.py --host 0.0.0.0 --port 8000 --model int8
"""

import os
import sys
import argparse
from pathlib import Path
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))


def main():
    parser = argparse.ArgumentParser(description="EdgePPE-Vision Inference Server Launcher")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Listening port (default: 8000)")
    parser.add_argument("--model", type=str, choices=["int8", "fp32"], default="int8", help="Active model type")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload for development")
    parser.add_argument("--threads", type=int, default=None, help="Inference CPU thread count")
    args = parser.parse_args()

    # Pass configuration to FastAPI app via environment variables
    os.environ["EDGEPPE_HOST"] = args.host
    os.environ["EDGEPPE_PORT"] = str(args.port)
    os.environ["EDGEPPE_MODEL"] = args.model
    if args.threads:
        os.environ["EDGEPPE_THREADS"] = str(args.threads)

    print("=" * 65)
    print("  EdgePPE-Vision: Quantized Edge AI PPE Compliance System")
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"  Interactive Swagger Docs: http://{args.host}:{args.port}/docs")
    print(f"  Live Demo Web Dashboard: http://{args.host}:{args.port}/")
    print(f"  Active Model: {args.model.upper()} Quantized Graph")
    print("=" * 65)

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
