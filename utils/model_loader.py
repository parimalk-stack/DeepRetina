import torch
import torch.nn as nn
import os
import sys
import logging
import timm
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Model metadata
MODEL_METADATA = {
    'cnn_model': {
        'name': 'Custom CNN',
        'description': 'Custom CNN with 4 convolutional blocks',
        'input_size': (1, 224, 224),
        'parameters': '~2.5M',
        'architecture': 'CNN'
    },
    'vit_model': {
        'name': 'Vision Transformer',
        'description': 'ViT-Base with patch size 16x16',
        'input_size': (3, 224, 224),
        'parameters': '~86M',
        'architecture': 'Transformer'
    },
    'swin_model': {
        'name': 'Swin Transformer',
        'description': 'Swin-Tiny with window size 7x7',
        'input_size': (3, 224, 224),
        'parameters': '~28M',
        'architecture': 'Transformer'
    }
}

# Define the CNN model class
class OCTModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(OCTModel, self).__init__()
        
        img_size = 224
        
        # First convolutional block
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25)
        )
        
        # Second convolutional block
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25)
        )
        
        # Third convolutional block
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25)
        )
        
        # Fourth convolutional block
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25)
        )
        
        # Calculate the size after all convolutions and pooling
        fc_input_size = 256 * (img_size // 16) * (img_size // 16)
        
        # Fully connected layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(fc_input_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x

# Define Swin Transformer model class
class SwinTransformerModel:
    def __init__(self, model_path, num_classes=4):
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
    def __call__(self, x):
        return self.model(x)
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def eval(self):
        self.model.eval()
        return self

def load_cnn_model(model_path):
    """Load the CNN model"""
    try:
        logger.info(f"Loading CNN model from {model_path}")
        model = OCTModel().to('cpu')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        logger.info("CNN model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading CNN model: {str(e)}")
        raise

def load_vit_model(model_path):
    """Load the ViT model"""
    try:
        logger.info(f"Loading ViT model from {model_path}")
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(model_path)
        model.eval()
        logger.info("ViT model loaded successfully")
        
        # Return both model and processor as a wrapper
        class ViTWrapper:
            def __init__(self, model, processor):
                self.model = model
                self.processor = processor
                
            def __call__(self, x):
                # x should be PIL Image for ViT
                if isinstance(x, torch.Tensor):
                    # Convert tensor to PIL Image
                    if x.dim() == 4:
                        x = x.squeeze(0)
                    if x.shape[0] == 3:  # RGB
                        x = x.permute(1, 2, 0)
                    x = (x * 255).clamp(0, 255).byte().numpy()
                    x = Image.fromarray(x)
                
                inputs = self.processor(x, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                return outputs.logits
            
            def to(self, device):
                self.model.to(device)
                return self
            
            def eval(self):
                self.model.eval()
                return self
        
        return ViTWrapper(model, processor)
        
    except Exception as e:
        logger.error(f"Error loading ViT model: {str(e)}")
        raise

def load_swin_model(model_path):
    """Load the Swin Transformer model"""
    try:
        logger.info(f"Loading Swin model from {model_path}")
        model = SwinTransformerModel(model_path)
        logger.info("Swin model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Swin model: {str(e)}")
        raise

def load_model(model_path, model_type='cnn_model'):
    """Unified model loading function"""
    try:
        if model_type == 'cnn_model':
            return load_cnn_model(model_path)
        elif model_type == 'vit_model':
            return load_vit_model(model_path)
        elif model_type == 'swin_model':
            return load_swin_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"Error loading model {model_type}: {str(e)}")
        raise

def get_model_metadata(model_type):
    """Get metadata for a specific model type"""
    return MODEL_METADATA.get(model_type, {})

def get_all_model_metadata():
    """Get metadata for all models"""
    return MODEL_METADATA

def preprocess_for_model(image, model_type):
    """Preprocess image according to model requirements"""
    try:
        if model_type == 'cnn_model':
            # CNN expects grayscale (1, H, W)
            if len(image.shape) == 3:
                if image.shape[0] == 3:  # RGB channels first
                    image = torch.mean(image, dim=0, keepdim=True)
                elif image.shape[2] == 3:  # RGB channels last
                    image = torch.mean(torch.from_numpy(image), dim=2, keepdim=True)
                    image = image.permute(2, 0, 1)
            elif len(image.shape) == 2:
                image = torch.from_numpy(image).unsqueeze(0)
            
            return image.unsqueeze(0)  # Add batch dimension
            
        elif model_type in ['vit_model', 'swin_model']:
            # ViT and Swin expect RGB (3, H, W)
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image, image, image], axis=0)
            elif len(image.shape) == 3:
                if image.shape[0] == 1:  # Single channel, channels first
                    image = np.repeat(image, 3, axis=0)
                elif image.shape[2] == 1:  # Single channel, channels last
                    image = image.squeeze(-1)
                    image = np.stack([image, image, image], axis=0)
                elif image.shape[2] == 3:  # RGB channels last
                    image = np.transpose(image, (2, 0, 1))
            
            if model_type == 'vit_model':
                # For ViT, we need PIL Image
                if isinstance(image, np.ndarray):
                    if image.shape[0] == 3:  # Channels first
                        image = np.transpose(image, (1, 2, 0))
                    image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                return image
            else:
                # For Swin, return as tensor
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                return image.unsqueeze(0)  # Add batch dimension
                
    except Exception as e:
        logger.error(f"Error preprocessing for {model_type}: {str(e)}")
        raise
