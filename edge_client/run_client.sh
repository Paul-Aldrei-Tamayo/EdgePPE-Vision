#!/bin/bash
# EdgePPE-Vision: Raspberry Pi Client Launcher
# Usage: ./run_client.sh <SERVER_IP> [SOURCE]
# Example: ./run_client.sh 192.168.1.100 0

SERVER_IP=${1:-"127.0.0.1"}
SOURCE=${2:-"0"}

echo "=================================================="
echo "  Launching EdgePPE-Vision Client on Raspberry Pi"
echo "  Inference Server: http://${SERVER_IP}:8000"
echo "  Camera Source   : ${SOURCE}"
echo "=================================================="

python3 pi_client.py --server "http://${SERVER_IP}:8000" --source "${SOURCE}" --width 640 --height 480 --fps 6.0
