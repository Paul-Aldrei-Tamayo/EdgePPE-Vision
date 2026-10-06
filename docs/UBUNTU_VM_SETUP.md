# Ubuntu 22.04 LTS Server VM Setup Guide (Oracle VirtualBox)

This guide walks through configuring the **Ubuntu 22.04 LTS Server Virtual Machine** on Oracle VirtualBox to host the **EdgePPE-Vision** FastAPI and ONNX Runtime inference microservice.

---

## 1. Virtual Machine Specifications

As specified in Proposal Section 7 & Table (Page 10):
- **Base OS**: Ubuntu 22.04 LTS (Server or Desktop)
- **vCPU Cores**: 2 vCPU cores (minimum)
- **RAM**: 4 GB (minimum 2 GB allocated)
- **Disk Storage**: 25 GB VDI (Dynamically allocated)
- **Virtualization Software**: Oracle VM VirtualBox 7.x

---

## 2. Critical Network Configuration: Bridged Adapter

To ensure the Raspberry Pi client can directly reach the FastAPI server over your local Wi-Fi or Ethernet network without complex NAT routing:

1. Open **Oracle VirtualBox Manager**.
2. Select your Ubuntu Server VM and click **Settings**.
3. Navigate to **Network** -> **Adapter 1**.
4. Change **Attached to** from `NAT` to **`Bridged Adapter`**.
5. In **Name**, select your host PC's active physical network adapter (e.g. Wi-Fi adapter or Realtek PCIe Ethernet).
6. Under **Advanced**, ensure:
   - **Promiscuous Mode**: `Allow All` (or `Allow VMs`)
   - **Cable Connected**: `Checked [x]`
7. Click **OK** and start the VM.

> **Why Bridged Adapter?**
> In Bridged mode, your router assigns your Ubuntu VM its own unique IP address (e.g., `192.168.1.150`) on the exact same subnet as your Raspberry Pi, allowing direct, ultra-low-latency HTTP communication.

---

## 3. Ubuntu Environment Setup

1. Log into your Ubuntu VM terminal.
2. Update packages:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. Install Python 3.10+, pip, and OpenCV system dependencies:
   ```bash
   sudo apt install -y python3-pip python3-venv git curl libgl1 libglib2.0-0
   ```

---

## 4. Deploying EdgePPE-Vision Server

1. Transfer or clone the repository to the Ubuntu VM:
   ```bash
   git clone <repo-url> ~/EdgePPE-Vision
   cd ~/EdgePPE-Vision/backend_server
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify models are present in `../models/`:
   ```bash
   ls -lh ../models/
   ```
   *(Ensure `yolov8n_ppe_int8.onnx` and `labels.json` exist).*

---

## 5. Launching the Inference Server

### Interactive Launch:
```bash
python3 run_server.py --host 0.0.0.0 --port 8000 --model int8
```

You should see:
```text
=================================================================
  EdgePPE-Vision: Quantized Edge AI PPE Compliance System
  Listening on: http://0.0.0.0:8000
  Interactive Swagger Docs: http://0.0.0.0:8000/docs
  Live Demo Web Dashboard: http://0.0.0.0:8000/
  Active Model: INT8 Quantized Graph
=================================================================
```

### Find Server IP Address:
In a new terminal on Ubuntu VM:
```bash
hostname -I
```
Take note of this IP (e.g., `192.168.1.150`). This is the IP you will provide to the Raspberry Pi client!

---

## 6. Testing the Server Locally

Open a browser on your host machine or VM:
- **Interactive Dashboard**: `http://<server-ip>:8000/`
- **Swagger Documentation**: `http://<server-ip>:8000/docs`
- **Health Check**: `http://<server-ip>:8000/api/v1/health`

---

## 7. Permanent Background Service (Systemd)

To ensure the FastAPI inference service runs 24/7 without needing an open terminal:

1. Create a systemd service file:
   ```bash
   sudo nano /etc/systemd/system/edgeppe-server.service
   ```
2. Add the following:
   ```ini
   [Unit]
   Description=EdgePPE-Vision FastAPI Inference Service
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/EdgePPE-Vision/backend_server
   ExecStart=/home/ubuntu/EdgePPE-Vision/backend_server/venv/bin/python3 run_server.py --host 0.0.0.0 --port 8000 --model int8
   Restart=always
   RestartSec=3

   [Install]
   WantedBy=multi-user.target
   ```
3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable edgeppe-server.service
   sudo systemctl start edgeppe-server.service
   ```
4. Verify service status:
   ```bash
   sudo systemctl status edgeppe-server.service
   ```
