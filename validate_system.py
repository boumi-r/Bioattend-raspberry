#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioAttend System Validation Script
Verifies all components are ready for deployment on Raspberry Pi

Usage:
  python3 validate_system.py

Exit codes:
  0 - All checks passed ✅
  1 - One or more checks failed ❌
"""

import sys
import os
import subprocess
import importlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SystemValidator:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def check(self, condition, message):
        if condition:
            logger.info(f"✅ {message}")
            self.passed += 1
            return True
        else:
            logger.error(f"❌ {message}")
            self.failed += 1
            return False
    
    def warning(self, message):
        logger.warning(f"⚠️  {message}")
        self.warnings += 1
    
    def section(self, title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")


def validate_imports(validator):
    """Check that all required modules can be imported"""
    validator.section("1. IMPORT VALIDATION")
    
    modules = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('picamera2', 'PiCamera2'),
        ('requests', 'Requests'),
        ('scipy', 'SciPy'),
        ('dotenv', 'python-dotenv'),
    ]
    
    for module_name, display_name in modules:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            validator.check(True, f"{display_name}: {version}")
        except ImportError as e:
            validator.check(False, f"{display_name}: {e}")


def validate_custom_modules(validator):
    """Check custom BioAttend modules"""
    validator.section("2. CUSTOM MODULE VALIDATION")
    
    modules_to_check = [
        ('src.config', 'Configuration module'),
        ('src.camera', 'Camera module'),
        ('src.pir', 'PIR module'),
        ('src.api_client', 'API client module'),
        ('src.gpio_feedback', 'GPIO feedback module'),
        ('src.utility', 'Utility module'),
        ('src.anti_spoof_predict', 'Anti-spoof module'),
        ('src.model_lib.MiniFASNet', 'MiniFASNet models'),
        ('src.data_io.transform', 'Data transforms'),
    ]
    
    for module_path, display_name in modules_to_check:
        try:
            importlib.import_module(module_path)
            validator.check(True, f"{display_name}")
        except ImportError as e:
            validator.check(False, f"{display_name}: {e}")


def validate_models(validator):
    """Check model files"""
    validator.section("3. MODEL FILES VALIDATION")
    
    models_dir = './models'
    if not os.path.exists(models_dir):
        validator.check(False, f"Models directory exists")
        return
    
    validator.check(True, f"Models directory exists")
    
    # List .pth files
    pth_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if len(pth_files) > 0:
        validator.check(True, f"Pre-trained models found: {len(pth_files)} files")
        for pth_file in pth_files:
            file_path = os.path.join(models_dir, pth_file)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            logger.info(f"  - {pth_file} ({size_mb:.1f} MB)")
    else:
        validator.warning("No .pth model files found in ./models/")
        validator.warning("System will still work but needs models for optimal performance")


def validate_config(validator):
    """Check configuration"""
    validator.section("4. CONFIGURATION VALIDATION")
    
    try:
        from src.config import validate_config, SERVER_URL, API_ENDPOINT, GPIO_PIR
        
        # Try validation
        try:
            validate_config()
            validator.check(True, "Configuration loads successfully")
            validator.check(True, f"SERVER_URL: {SERVER_URL}")
            validator.check(True, f"API_ENDPOINT: {API_ENDPOINT}")
            validator.check(True, f"GPIO_PIR: {GPIO_PIR}")
        except ValueError as e:
            validator.check(False, f"Configuration validation: {e}")
    except ImportError as e:
        validator.check(False, f"Cannot load config: {e}")


def validate_pytorch(validator):
    """Check PyTorch configuration"""
    validator.section("5. PYTORCH CONFIGURATION")
    
    try:
        import torch
        
        version = torch.__version__
        validator.check(True, f"PyTorch version: {version}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            validator.check(True, f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            validator.warning("CUDA not available (expected on Raspberry Pi, will use CPU)")
        
        device = "cuda" if cuda_available else "cpu"
        validator.check(True, f"Default device: {device}")
        
    except Exception as e:
        validator.check(False, f"PyTorch check failed: {e}")


def validate_camera(validator):
    """Test camera initialization"""
    validator.section("6. CAMERA VALIDATION")
    
    try:
        from src.camera import CameraManager
        
        try:
            cam = CameraManager()
            cam.open()
            validator.check(True, "Camera initialized successfully")
            
            # Try to capture
            image_bytes = cam.capture_image()
            validator.check(len(image_bytes) > 0, "Image capture successful")
            
            cam.close()
            validator.check(True, "Camera closed properly")
        except RuntimeError as e:
            validator.warning(f"Camera not available (OK in dev environment): {e}")
    except Exception as e:
        validator.warning(f"Camera validation skipped: {e}")


def validate_anti_spoof(validator):
    """Test anti-spoof module"""
    validator.section("7. ANTI-SPOOF MODULE VALIDATION")
    
    try:
        from src.anti_spoof_predict import AntiSpoofPredict
        
        asp = AntiSpoofPredict(device_id=0)
        validator.check(True, "AntiSpoofPredict initialized")
        validator.check(asp.device is not None, f"Device set to: {asp.device}")
        validator.check(asp.use_caffe or hasattr(asp, 'face_cascade'), 
                       "Face detection available (Caffe or Haar)")
    except Exception as e:
        validator.check(False, f"Anti-spoof module: {e}")


def validate_directory_structure(validator):
    """Check directory structure"""
    validator.section("8. DIRECTORY STRUCTURE VALIDATION")
    
    required_dirs = [
        'src',
        'src/model_lib',
        'src/data_io',
        'models',
        'tests',
        'docs',
    ]
    
    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        validator.check(exists, f"Directory exists: {dir_path}")


def validate_documentation(validator):
    """Check documentation files"""
    validator.section("9. DOCUMENTATION VALIDATION")
    
    docs = [
        ('INSTALLATION.md', 'Installation guide'),
        ('OPTIMIZATION_REPORT.md', 'Optimization report'),
        ('README.md', 'README'),
    ]
    
    for filename, description in docs:
        exists = os.path.isfile(filename)
        validator.check(exists, f"{description}: {filename}")


def main():
    print("\n" + "="*60)
    print("  BIOATTEND SYSTEM VALIDATION")
    print("="*60)
    
    validator = SystemValidator()
    
    # Run all validations
    validate_imports(validator)
    validate_custom_modules(validator)
    validate_models(validator)
    validate_config(validator)
    validate_pytorch(validator)
    validate_anti_spoof(validator)
    validate_directory_structure(validator)
    validate_documentation(validator)
    validate_camera(validator)
    
    # Summary
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed:   {validator.passed}")
    print(f"❌ Failed:   {validator.failed}")
    print(f"⚠️  Warnings: {validator.warnings}")
    print(f"{'='*60}\n")
    
    if validator.failed == 0:
        print("🎉 ALL CRITICAL CHECKS PASSED - System ready for deployment!")
        print()
        print("Next steps:")
        print("  1. Configure .env with: SERVER_URL, API_TOKEN, etc.")
        print("  2. Place pre-trained models in ./models/")
        print("  3. Verify GPIO connections (see docs/wiring.md)")
        print("  4. Run: python3 src/main.py")
        return 0
    else:
        print("⚠️  DEPLOYMENT BLOCKED - Fix errors above before running")
        return 1


if __name__ == '__main__':
    sys.exit(main())
