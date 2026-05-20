"""Visualization utilities for OCT analysis results.

This module provides functions for creating visualizations of OCT analysis results,
including prediction probabilities and class distributions.
"""

import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def create_visualization(original_img, processed_img, predicted_class, probabilities, output_path):
    """Create a visualization of the analysis results.
    
    Args:
        original_img (numpy.ndarray): Original input image
        processed_img (numpy.ndarray): Preprocessed image
        predicted_class (str): Predicted class label
        probabilities (dict): Dictionary of class probabilities
        output_path (str): Path to save the visualization
        
    Returns:
        str: Path to the saved visualization
        
    Raises:
        ValueError: If visualization creation fails
    """
    try:
        # Convert original image to RGB for display
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Display original image
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display processed image
        ax2.imshow(processed_img, cmap='gray')
        ax2.set_title('Processed Image')
        ax2.axis('off')
        
        # Create probability bar chart
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        bars = ax3.bar(classes, probs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('Class Probabilities')
        ax3.set_ylabel('Probability')
        ax3.set_ylim(0, 1)
        
        # Add probability values on top of bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # Add prediction summary text
        ax4.axis('off')
        summary_text = f"""PREDICTION SUMMARY

Predicted Class: {predicted_class}
Confidence: {probabilities.get(predicted_class, 0):.1%}

Class Probabilities:
"""
        
        for cls, prob in probabilities.items():
            summary_text += f"• {cls}: {prob:.1%}\n"
            
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        logger.info(f"Created visualization at {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {str(e)}")
        raise ValueError(f"Failed to create visualization: {str(e)}") from e
