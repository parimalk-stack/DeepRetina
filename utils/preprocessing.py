"""Image preprocessing utilities for OCT analysis.

This module provides functions for preprocessing OCT images, including
normalization, contrast enhancement, and resizing.
"""

import os
import logging
import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_image(image, output_dir):
    """Preprocess an OCT image for analysis.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        output_dir (str): Directory to save preprocessed image
        
    Returns:
        tuple: (preprocessed_image, output_path)
        
    Raises:
        ValueError: If image is invalid or preprocessing fails
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Converted image to grayscale")
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        logger.info("Applied CLAHE contrast enhancement")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
        logger.info("Applied Gaussian blur")
        
        # Resize image to standard size
        resized = cv2.resize(blurred, (224, 224))
        logger.info("Resized image to 224x224")
        
        # Save preprocessed image
        output_path = os.path.join(output_dir, 'preprocessed.png')
        cv2.imwrite(output_path, resized)
        logger.info(f"Saved preprocessed image to {output_path}")
        
        return resized, output_path
        
    except cv2.error as e:
        logger.error(f"OpenCV error during preprocessing: {str(e)}")
        raise ValueError(f"Image preprocessing failed: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {str(e)}")
        raise ValueError(f"Image preprocessing failed: {str(e)}") from e
