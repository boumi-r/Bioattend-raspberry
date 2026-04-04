# BioAttend Raspberry Pi 5 - Setup Guide

## System Overview
BioAttend is a real-time access control system optimized for Raspberry Pi 5 that combines:
- PIR motion detection
- Anti-spoofing face liveness verification (MiniFASNet)
- Django server integration for identity verification
- GPIO feedback (LED/buzzer/screen messages)

## Pre-Installation Requirements

### Hardware
- Raspberry Pi 5 (4GB RAM minimum, 8GB recommended)
- PiCamera 3 or equivalent with CSI connector
- PIR motion sensor (HC-SR501 or similar)
- Power supply (27W USB-C recommended)
- MicroSD card (32GB or larger, Class 10 recommended)

### System Setup
1. **Raspberry Pi OS (Bookworm 64-bit)**
   ```bash
   # Ensure OS is updated
   sudo apt update && sudo apt upgrade -y
   ```

2. **System Dependencies**
   ```bash
   sudo apt install -y \
     python3-dev \
     python3-pip \
     build-essential \
     cmake \
     git \
     libatlas-base-dev \
     libjasper-dev \
     libtiff-dev \
     libjasper1 \
     libharfbuzz0b \
     libwebp6 \
     libtiff5 \
     libjasper-dev \
     libhdf5-dev \
     libharfbuzz0b \
     libwebp6
   ```

## Installation Steps

### 1. Clone Repository
```bash
cd ~
git clone https://github.com/boumi-r/Bioattend-raspberry.git
cd Bioattend-raspberry
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install -r requirements.txt

# Special handling for PyTorch on Raspberry Pi
# Option A: Pre-built wheels (recommended)
pip install torch torchvision --break-system-packages

# Option B: Build from source (slower but more compatible)
# See: https://github.com/nmilosev/pytorch-raspberry-pi-os
```

### 4. Configure Environment Variables
```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Key parameters to set:**
```
SERVER_URL=http://192.168.1.100:8000        # Django server address
API_ENDPOINT=http://192.168.1.100:8000/api/face/analyze/
API_TOKEN=your_secret_token                 # Optional authentication
GPIO_PIR=17                                 # PIR sensor GPIO pin
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
DEBUG=True                                  # Set to False in production
```

### 5. Prepare Pre-trained Models
Ensure the following models are in `./models/`:
- `2.7_80x80_MiniFASNetV2.pth` (primary)
- `4_0_0_80x80_MiniFASNetV1SE.pth` (backup)

**Note:** If Caffe detection model files are available, place them in:
- `./resources/detection_model/Widerface-RetinaFace.caffemodel`
- `./resources/detection_model/deploy.prototxt`

Otherwise, the system automatically falls back to Haar cascades.

### 6. Test Installation
```bash
# Test imports
python3 -c "
import cv2
import torch
import picamera2
from src.anti_spoof_predict import AntiSpoofPredict
print('✓ All imports successful')
"

# Test camera
python3 -c "
from src.camera import CameraManager
cam = CameraManager()
cam.open()
print('✓ Camera initialized')
cam.close()
"
```

## Running the System

### Foreground (Debugging)
```bash
source venv/bin/activate
python3 src/main.py
```

### Background (systemd Service - Production)

**1. Create systemd service file:**
```bash
sudo nano /etc/systemd/system/bioattend.service
```

**2. Insert content:**
```
[Unit]
Description=BioAttend Access Control System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Bioattend-raspberry
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/pi/Bioattend-raspberry/venv/bin/python3 src/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**3. Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable bioattend
sudo systemctl start bioattend
```

**4. Monitor logs:**
```bash
sudo journalctl -u bioattend -f
```

## Performance Optimization Tips

### 1. Disable GUI (Headless Mode)
```bash
sudo raspi-config
# Select: 1 System Options > S5 Boot / Auto Login > B2 Console
```

### 2. Reduce Resolution if Needed
```env
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
```

### 3. Enable GPU Memory Split
```bash
sudo nano /boot/firmware/config.txt
# Add: gpu_mem=128
```

### 4. Tune Kernel Parameters
```bash
sudo nano /etc/sysctl.conf
# Add:
vm.swappiness=10
```

## Troubleshooting

### PyTorch Installation Fails
```bash
# Use pre-built wheel
wget https://files.pythonhosted.org/packages/.../torch-...whl
pip install ./torch-...whl
```

### Camera Not Detected
```bash
# Check camera connection
libcamera-hello --list-cameras

# Verify dtoverlay
cat /boot/firmware/config.txt | grep dtoverlay
```

### PIR Sensor Not Triggering
```bash
# Test GPIO manually
python3 -c "
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)  # Adjust pin as needed
while True:
    print('PIR:', GPIO.input(17))
"
```

### Django Server Connection Issues
```bash
# Test connectivity
curl -v http://192.168.1.100:8000/api/face/analyze/
```

## Architecture Summary

```
PIR Sensor
    ↓ (motion detected)
Camera Capture (PiCamera2)
    ↓
Image Processing (OpenCV)
    ↓
Anti-Spoof Detection (MiniFASNet on CPU)
    ├─ Face Detection (Haar Cascade or Caffe)
    └─ Liveness Classification (Real vs Fake)
    ↓
HTTP POST to Django Server
    ├─ Face Detection (InsightFace)
    ├─ Embedding Extraction (512-dim vector)
    └─ Database Comparison (Cosine Distance)
    ↓
Access Decision + Feedback
    ├─ GPIO Signals (LED/Buzzer)
    └─ Screen Messages
```

## Files Structure
```
- src/
  ├── main.py                   # Main loop (PIR → Detection → API)
  ├── anti_spoof_predict.py     # MiniFASNet anti-spoofing
  ├── camera.py                 # PiCamera2 wrapper
  ├── pir.py                    # PIR sensor handler
  ├── api_client.py             # Django API interface
  ├── gpio_feedback.py          # Feedback display
  ├── config.py                 # Configuration loader
  ├── utility.py                # Helper functions
  ├── model_lib/                # Neural network models
  │   └── MiniFASNet.py
  └── data_io/                  # Data preprocessing
      └── transform.py
- models/                        # Pre-trained .pth files
- tests/                         # Unit tests
- docs/                          # Documentation
```

## Security Notes

1. **API Token:** Always set a strong API_TOKEN in .env
2. **Network:** Use HTTPS in production (configure reverse proxy)
3. **Credentials:** Never commit .env to version control
4. **Model Files:** Keep .pth files secure (they contain sensitive data)

## Performance Metrics (Raspberry Pi 5)

- **Model Loading:** ~2-3 seconds
- **Face Detection:** ~50-100ms (Haar) / ~200-300ms (Caffe)
- **Anti-Spoof Prediction:** ~150-250ms
- **Total Pipeline:** ~400-600ms per frame
- **FPS:** ~2-3 face detections/second
- **Memory Usage:** ~600-800MB

## Next Steps

1. Deploy Django server (if not already running)
2. Configure employee database with face embeddings
3. Set up environment-specific .env files
4. Run comprehensive tests on actual hardware
5. Monitor logs for 24-48 hours in production

## Support

For issues, check:
- Logs: `bioattend.log`
- Documentation: `docs/wiring.md` (GPIO connections)
- GitHub Issues: https://github.com/boumi-r/Bioattend-raspberry

## License
See repository for license terms.
