# Raspberry Pi 4 / 5 Setup Guide: EdgePPE-Vision Edge Client

This guide details the step-by-step setup and deployment instructions for configuring a **Raspberry Pi 4 or Raspberry Pi 5** as an Edge Client node in the **EdgePPE-Vision** distributed architecture.

---

## 1. Hardware Requirements

- **Raspberry Pi**: Raspberry Pi 4 Model B (2GB, 4GB, or 8GB) OR Raspberry Pi 5.
- **Camera Options**:
  - **Option A (Recommended)**: Raspberry Pi Camera Module v2 / v3 / HQ Camera connected via ribbon cable to the CSI port (`CAM/DISP`).
  - **Option B**: Standard USB UVC Webcam plugged into any USB 3.0 (blue) port.
  - **Option C**: Video simulation (using pre-recorded mp4 files in `samples/videos/`).
- **Storage**: MicroSD Card (Class 10, A1/A2, minimum 16 GB).
- **Power Supply**: Official 5V 3A (for Pi 4) or 5V 5A USB-C PD (for Pi 5).
- **Network**: Connected to the same Wi-Fi router or Ethernet switch as the Ubuntu Server VM.
- **Display (Optional)**: Micro-HDMI to HDMI monitor for live bounding box visualization.

---

## 2. Raspberry Pi OS Installation & Preparation

1. Flash the SD Card with **Raspberry Pi OS (64-bit)** using the [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Enable SSH, configure Wi-Fi credentials, and set your username/password during flashing.
3. Boot the Raspberry Pi and open a terminal (or SSH into it):
   ```bash
   ssh pi@<raspberry-pi-ip>
   ```
4. Update system packages:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
5. Install required system dependencies:
   ```bash
   sudo apt install -y python3-pip python3-venv python3-opencv libgl1-mesa-glx git
   ```

---

## 3. Camera Setup & Verification

### For Raspberry Pi Camera Module (CSI Ribbon Cable):
1. Shut down the Pi:
   ```bash
   sudo shutdown -h now
   ```
2. Insert the ribbon cable into the camera port:
   - On **Pi 4**: Pull up the clip on the `CAMERA` port, insert ribbon cable with blue backing facing the Ethernet port, then push clip down.
   - On **Pi 5**: Use the dedicated mini CSI/DSI adapter ribbon cable (`CAM/DISP 0` or `CAM/DISP 1`), contacts facing inward.
3. Boot up and verify camera detection:
   ```bash
   rpicam-hello -t 3000
   ```
   *(If a 3-second live camera window appears, the CSI camera is working perfectly).*

### For USB Webcam:
1. Plug the webcam into a USB port.
2. Verify detection:
   ```bash
   v4l2-ctl --list-devices
   # or
   ls -l /dev/video*
   ```
   The device will usually appear at `/dev/video0`.

---

## 4. Edge Client Software Installation

1. Copy the `edge_client` directory and `samples` to your Raspberry Pi:
   ```bash
   scp -r edge_client samples pi@<raspberry-pi-ip>:~/edgeppe_client/
   ```
   *(Or clone the repository directly onto the Pi)*.

2. On the Raspberry Pi, navigate to the folder:
   ```bash
   cd ~/edgeppe_client/edge_client
   ```

3. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv --system-site-packages
   source venv/bin/activate
   ```
   *(Note: `--system-site-packages` allows access to system-optimized `picamera2` and `libcamera` bindings).*

4. Install client Python dependencies:
   ```bash
   pip install -r requirements_pi.txt
   ```

---

## 5. Running the Edge Client

### Step 5.1: Obtain the Server IP
Make sure your FastAPI server is running on the Ubuntu VM / host PC.
On your server, run:
```bash
hostname -I
```
*(Example: `192.168.1.150`)*.

### Step 5.2: Test Connectivity
On the Raspberry Pi:
```bash
python3 test_client.py --server http://192.168.1.150:8000
```
If you see `HTTP Status Code: 200`, your connection is verified!

### Step 5.3: Launch Real-Time Monitoring

**Using Raspberry Pi Camera Module:**
```bash
python3 pi_client.py --server http://192.168.1.150:8000 --source 0 --fps 6.0
```

**Using USB Webcam:**
```bash
python3 pi_client.py --server http://192.168.1.150:8000 --source 0
```

**Using Pre-recorded Video Clip (Demo / Test Mode):**
```bash
python3 pi_client.py --server http://192.168.1.150:8000 --source ../samples/videos/ppe-1.mp4
```

**Running Headless (Over SSH with no HDMI monitor):**
```bash
python3 pi_client.py --server http://192.168.1.150:8000 --source 0 --headless
```

---

## 6. Client Keyboard Shortcuts

When the visual display window is active:
- **`q`** : Quit the client application gracefully.
- **`s`** : Save timestamped visual snapshot to `snapshots/` folder.
- **`c`** : Cycle confidence threshold (`0.25` -> `0.35` -> `0.45` -> `0.55`).
- **`h`** : Toggle telemetry HUD overlay on/off.

---

## 7. Autostart on Boot (Systemd Service)

To run EdgePPE-Vision automatically when the Raspberry Pi boots:

1. Create a service file:
   ```bash
   sudo nano /etc/systemd/system/edgeppe-client.service
   ```
2. Paste the following configuration (replace `<SERVER_IP>` and username):
   ```ini
   [Unit]
   Description=EdgePPE-Vision Raspberry Pi Edge Client
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/edgeppe_client/edge_client
   ExecStart=/home/pi/edgeppe_client/edge_client/venv/bin/python3 pi_client.py --server http://<SERVER_IP>:8000 --source 0 --fps 6.0 --headless
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable edgeppe-client.service
   sudo systemctl start edgeppe-client.service
   ```
4. Check status:
   ```bash
   sudo systemctl status edgeppe-client.service
   ```
