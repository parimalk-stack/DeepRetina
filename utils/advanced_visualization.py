# utils/advanced_visualization.py - Advanced Visualization Module for OCT Analysis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")

class OCTVisualization:
    """Advanced visualization class for OCT analysis results"""
    
    def __init__(self):
        self.class_colors = {
            'CNV': '#FF6B6B',      # Red
            'DME': '#4ECDC4',      # Teal
            'DRUSEN': '#45B7D1',   # Blue
            'NORMAL': '#96CEB4'    # Green
        }
        
    def create_result_visualization(self, original_image, predicted_class, confidence, 
                                  all_probabilities, processing_time, save_path=None):
        """Create comprehensive result visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'OCT Analysis Results - {predicted_class}', fontsize=16, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title('Original OCT Image')
            axes[0, 0].axis('off')
            
            # Prediction confidence pie chart
            classes = list(all_probabilities.keys())
            probs = list(all_probabilities.values())
            colors = [self.class_colors[cls] for cls in classes]
            
            wedges, texts, autotexts = axes[0, 1].pie(probs, labels=classes, autopct='%1.1f%%', 
                                                     colors=colors, startangle=90)
            axes[0, 1].set_title('Class Probabilities')
            
            # Highlight predicted class
            max_idx = np.argmax(probs)
            wedges[max_idx].set_edgecolor('black')
            wedges[max_idx].set_linewidth(3)
            
            # Confidence bar chart
            bars = axes[1, 0].bar(classes, probs, color=colors, alpha=0.8)
            axes[1, 0].set_title('Confidence Scores')
            axes[1, 0].set_ylabel('Probability')
            axes[1, 0].set_ylim(0, 1)
            
            # Highlight predicted class
            bars[max_idx].set_edgecolor('black')
            bars[max_idx].set_linewidth(3)
            
            # Add value labels
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom')
            
            # Summary information
            axes[1, 1].axis('off')
            summary_text = f"""PREDICTION SUMMARY

Predicted Class: {predicted_class}
Confidence: {confidence:.1%}
Processing Time: {processing_time:.3f}s

TOP PREDICTIONS:
"""
            
            # Sort probabilities
            sorted_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
            for i, (cls, prob) in enumerate(sorted_probs[:3]):
                summary_text += f"{i+1}. {cls}: {prob:.1%}\n"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating result visualization: {str(e)}")
            return None

    def create_quality_assessment_visualization(self, quality_result, save_path=None):
        """Create visualization for quality assessment results"""
        try:
            # Assuming quality_result contains 'original_image_path' and 'overall_quality_score'
            image_path = quality_result.get('original_image_path')
            quality_score = quality_result.get('overall_quality_score', 0)

            if not image_path or not os.path.exists(image_path):
                logger.error(f"Original image not found for quality visualization: {image_path}")
                return None

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Failed to load image for quality visualization: {image_path}")
                return None

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Quality Score: {quality_score:.2f}')
            ax.axis('off')

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                logger.info(f"Quality visualization saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating quality assessment visualization: {str(e)}")
            return None

def create_visualization_suite(results, output_dir):
    """Create complete visualization suite for OCT analysis results"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        viz = OCTVisualization()
        created_files = []
        
        # Single result visualization
        if 'original_image' in results:
            single_fig = viz.create_result_visualization(
                results['original_image'],
                results['predicted_class'],
                results['confidence'],
                results['all_probabilities'],
                results.get('processing_time', 0)
            )
            if single_fig:
                single_path = os.path.join(output_dir, 'single_result.png')
                single_fig.savefig(single_path, dpi=300, bbox_inches='tight')
                created_files.append(single_path)
                plt.close(single_fig)
        
        return {
            'success': True,
            'created_files': created_files,
            'output_directory': output_dir
        }
        
    except Exception as e:
        logger.error(f"Error creating visualization suite: {str(e)}")
        return {'error': str(e)}
