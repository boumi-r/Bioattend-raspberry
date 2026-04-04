# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm
# @Modified : Optimized for Raspberry Pi with fallback face detection

import os
import cv2
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class Detection:
    """Face detection with fallback to Haar cascades for Raspberry Pi compatibility"""
    def __init__(self):
        self.use_caffe = False
        self.detector = None
        self.detector_confidence = 0.6
        
        # Try to load Caffe model, but fallback to Haar cascades
        try:
            caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
            deploy = "./resources/detection_model/deploy.prototxt"
            if os.path.exists(caffemodel) and os.path.exists(deploy):
                self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
                self.use_caffe = True
                logger.info("Using Caffe face detector")
            else:
                raise FileNotFoundError("Caffe model files not found")
        except (FileNotFoundError, cv2.error) as e:
            logger.warning(f"Caffe detector failed ({e}), using Haar cascades (Pi-optimized)")
            self.use_caffe = False
            # Load Haar cascades
            opencv_data = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(
                os.path.join(opencv_data, "haarcascade_frontalface_default.xml")
            )

    def get_bbox(self, img):
        """Get bounding box using available detector"""
        if self.use_caffe and self.detector:
            return self._get_bbox_caffe(img)
        else:
            return self._get_bbox_haar(img)
    
    def _get_bbox_caffe(self, img):
        """Get bbox using Caffe RetinaFace detector"""
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img_resized = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        blob = cv2.dnn.blobFromImage(img_resized, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        
        if len(out.shape) == 1:
            out = out.reshape(1, -1)
        
        if out.shape[0] == 0:
            return None
            
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = (out[max_conf_index, 3]*width, 
                                   out[max_conf_index, 4]*height,
                                   out[max_conf_index, 5]*width, 
                                   out[max_conf_index, 6]*height)
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

    def _get_bbox_haar(self, img):
        """Get bbox using Haar cascade (Pi-optimized, lightweight)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        face = max(faces, key=lambda f: f[2]*f[3])
        x, y, w, h = face
        return [x, y, w, h]


class AntiSpoofPredict(Detection):
    """Anti-spoofing prediction using MiniFASNet neural network"""
    def __init__(self, device_id=0):
        super(AntiSpoofPredict, self).__init__()
        # Use CPU by default on Raspberry Pi (more stable)
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and device_id >= 0:
            self.device = torch.device(f"cuda:{device_id}")
        
        logger.info(f"AntiSpoofPredict initialized on device: {self.device}")
        self.model = None
        self.model_path = None

    def _load_model(self, model_path):
        """Load pre-trained anti-spoofing model"""
        if self.model_path == model_path and self.model is not None:
            # Model already loaded
            return
        
        # Define model based on filename
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)
        
        logger.info(f"Loading model: {model_type} ({h_input}x{w_input})")
        
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # Load model weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            keys = iter(state_dict)
            first_layer_name = keys.__next__()
            
            # Handle DataParallel wrapper
            if first_layer_name.find('module.') >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]
                    new_state_dict[name_key] = value
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
            
            self.model_path = model_path
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, img, model_path):
        """
        Predict if face is real (live) or fake (spoofed)
        
        Args:
            img: OpenCV image (BGR format)
            model_path: Path to .pth model file
        
        Returns:
            numpy array [fake_score, real_score] with softmax applied
        """
        try:
            # Load model if needed
            self._load_model(model_path)
            
            # Prepare image - crop face region
            bbox = self.get_bbox(img)
            if bbox is None:
                logger.warning("No face detected in image")
                # Return neutral scores
                return np.array([[0.5, 0.5]])
            
            x, y, w, h = bbox
            # Add padding
            pad = int(max(w, h) * 0.1)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + 2*pad)
            h = min(img.shape[0] - y, h + 2*pad)
            
            face_img = img[y:y+h, x:x+w]
            
            if face_img.size == 0:
                logger.warning("Face region is empty")
                return np.array([[0.5, 0.5]])
            
            # Preprocess image
            test_transform = trans.Compose([
                trans.Resize((80, 80)),
                trans.ToTensor(),
            ])
            face_tensor = test_transform(face_img)
            
            # Add batch dimension
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                result = self.model.forward(face_tensor)
                result = F.softmax(result, dim=1).cpu().numpy()
            
            logger.debug(f"Anti-spoof prediction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result











