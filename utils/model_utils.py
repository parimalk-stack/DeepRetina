"""Model utilities for OCT analysis.

This module provides functions for loading and using machine learning models
for OCT image analysis, including model loading, prediction, and quality assessment.
"""

import os
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class OCTModel(nn.Module):
    """Custom model for OCT image analysis."""
    
    def __init__(self, num_classes=4):
        """Initialize the OCT model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_model(model_path):
    """Load a trained model from disk.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        torch.nn.Module: Loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OCTModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}") from e

def predict_image(model, image_path, device=None):
    """Make predictions on an image using the model.
    
    Args:
        model (torch.nn.Module): Loaded model
        image_path (str): Path to input image
        device (torch.device, optional): Device to run inference on
        
    Returns:
        tuple: (predicted_class, class_probabilities)
        
    Raises:
        ValueError: If prediction fails
    """
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            
        # Convert to dictionary
        class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        class_probs = {name: prob.item() for name, prob in zip(class_names, probabilities)}
        
        logger.info(f"Prediction completed for {image_path}")
        return class_names[predicted_class], class_probs
        
    except Exception as e:
        logger.error(f"Failed to make prediction: {str(e)}")
        raise ValueError(f"Failed to make prediction: {str(e)}") from e

def assess_image_quality(image_path):
    """Assess the quality of an OCT image.
    
    Args:
        image_path (str): Path to input image
        
    Returns:
        dict: Quality assessment metrics
        
    Raises:
        ValueError: If quality assessment fails
    """
    try:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        # Calculate metrics
        metrics = {
            'brightness': np.mean(image),
            'contrast': np.std(image),
            'sharpness': cv2.Laplacian(image, cv2.CV_64F).var(),
            'noise': estimate_noise(image)
        }
        
        # Determine quality score
        quality_score = calculate_quality_score(metrics)
        metrics['quality_score'] = quality_score
        
        logger.info(f"Quality assessment completed for {image_path}")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to assess image quality: {str(e)}")
        raise ValueError(f"Failed to assess image quality: {str(e)}") from e

def estimate_noise(image):
    """Estimate noise level in an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        float: Estimated noise level
    """
    try:
        # Apply median filter
        median = cv2.medianBlur(image, 3)
        # Calculate difference
        diff = cv2.absdiff(image, median)
        # Return noise estimate
        return np.mean(diff)
    except Exception as e:
        logger.error(f"Failed to estimate noise: {str(e)}")
        return 0.0

def calculate_quality_score(metrics):
    """Calculate overall quality score from metrics.
    
    Args:
        metrics (dict): Dictionary of quality metrics
        
    Returns:
        float: Quality score between 0 and 1
    """
    try:
        # Normalize metrics
        brightness_score = min(metrics['brightness'] / 128, 1.0)
        contrast_score = min(metrics['contrast'] / 64, 1.0)
        sharpness_score = min(metrics['sharpness'] / 1000, 1.0)
        noise_score = max(1.0 - metrics['noise'] / 10, 0.0)
        
        # Calculate weighted average
        weights = {'brightness': 0.3, 'contrast': 0.3, 'sharpness': 0.2, 'noise': 0.2}
        score = (
            weights['brightness'] * brightness_score +
            weights['contrast'] * contrast_score +
            weights['sharpness'] * sharpness_score +
            weights['noise'] * noise_score
        )
        
        return min(max(score, 0.0), 1.0)
        
    except Exception as e:
        logger.error(f"Failed to calculate quality score: {str(e)}")
        return 0.0 