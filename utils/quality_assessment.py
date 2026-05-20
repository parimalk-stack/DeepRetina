# utils/quality_assessment.py - Quality Assessment Module for OCT Images

import cv2
import numpy as np
from PIL import Image
import logging
from scipy import ndimage
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import os

# Configure logging
logger = logging.getLogger(__name__)

# Quality assessment thresholds
QUALITY_THRESHOLDS = {
    'snr_threshold': 5.0,
    'motion_artifact_threshold': 0.15,
    'saturation_high_threshold': 0.2,
    'saturation_low_threshold': 0.3,
    'contrast_threshold': 0.1,
    'blur_threshold': 100.0
}

def calculate_snr(image):
    """Calculate Signal-to-Noise Ratio of OCT image"""
    try:
        # Define signal region (central retina)
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        signal_region = image[center_y-50:center_y+50, center_x-100:center_x+100]
        
        # Define noise region (background above retina)
        noise_region = image[0:50, :]
        
        # Calculate SNR
        signal_mean = np.mean(signal_region)
        noise_std = np.std(noise_region)
        
        if noise_std > 0:
            snr = 20 * np.log10(signal_mean / noise_std)
        else:
            snr = float('inf')
            
        return snr
    except Exception as e:
        logger.error(f"Error calculating SNR: {str(e)}")
        return 0.0

def detect_motion_artifacts(image):
    """Detect motion artifacts in OCT image"""
    try:
        # Horizontal gradient to detect vertical motion
        sobel_h = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        
        # Apply threshold to find strong horizontal edges
        threshold = 0.5 * np.std(sobel_h)
        strong_edges = np.abs(sobel_h) > threshold
        
        # Detect horizontal lines (potential motion artifacts)
        kernel = np.ones((1, 20), np.uint8)  # Horizontal kernel
        horizontal_lines = cv2.morphologyEx(strong_edges.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Count and locate horizontal artifacts
        num_artifacts = cv2.countNonZero(horizontal_lines) / (horizontal_lines.shape[0] * horizontal_lines.shape[1])
        
        return {
            'motion_artifact_score': num_artifacts,
            'has_significant_artifacts': num_artifacts > QUALITY_THRESHOLDS['motion_artifact_threshold'],
            'artifact_mask': horizontal_lines
        }
    except Exception as e:
        logger.error(f"Error detecting motion artifacts: {str(e)}")
        return {
            'motion_artifact_score': 0.0,
            'has_significant_artifacts': False,
            'artifact_mask': None
        }

def calculate_contrast(image):
    """Calculate image contrast using RMS contrast"""
    try:
        mean_intensity = np.mean(image)
        rms_contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
        normalized_contrast = rms_contrast / mean_intensity if mean_intensity > 0 else 0
        return normalized_contrast
    except Exception as e:
        logger.error(f"Error calculating contrast: {str(e)}")
        return 0.0

def detect_saturation(image):
    """Detect over/under saturation in image"""
    try:
        # For 8-bit images
        if image.dtype == np.uint8:
            over_saturated = np.mean(image > 250)
            under_saturated = np.mean(image < 5)
        else:
            # For normalized images [0, 1]
            over_saturated = np.mean(image > 0.98)
            under_saturated = np.mean(image < 0.02)
        
        return {
            'over_saturated_ratio': over_saturated,
            'under_saturated_ratio': under_saturated,
            'has_saturation_issues': (
                over_saturated > QUALITY_THRESHOLDS['saturation_high_threshold'] or 
                under_saturated > QUALITY_THRESHOLDS['saturation_low_threshold']
            )
        }
    except Exception as e:
        logger.error(f"Error detecting saturation: {str(e)}")
        return {
            'over_saturated_ratio': 0.0,
            'under_saturated_ratio': 0.0,
            'has_saturation_issues': False
        }

def calculate_blur_metric(image):
    """Calculate blur metric using Laplacian variance"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_metric = laplacian.var()
        
        return blur_metric
    except Exception as e:
        logger.error(f"Error calculating blur metric: {str(e)}")
        return 0.0

def analyze_retinal_layers(image):
    """Analyze retinal layer visibility and structure"""
    try:
        # Calculate horizontal intensity profile
        profile = np.mean(image, axis=1)
        
        # Find peaks (potential layer boundaries)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(profile, height=np.mean(profile))
        
        # Calculate layer visibility metrics
        layer_contrast = np.std(profile)
        layer_definition = len(peaks)
        
        return {
            'layer_contrast': layer_contrast,
            'layer_definition_score': layer_definition,
            'intensity_profile': profile,
            'detected_peaks': peaks
        }
    except Exception as e:
        logger.error(f"Error analyzing retinal layers: {str(e)}")
        return {
            'layer_contrast': 0.0,
            'layer_definition_score': 0,
            'intensity_profile': None,
            'detected_peaks': []
        }

def assess_image_quality_comprehensive(image_path):
    """Comprehensive quality assessment of OCT image"""
    try:
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {'error': 'Could not load image', 'processable': False}
        else:
            img = image_path
        
        # Ensure image is in correct format
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Basic metrics
        snr = calculate_snr(img)
        motion_artifacts = detect_motion_artifacts(img)
        contrast = calculate_contrast(img)
        saturation = detect_saturation(img)
        blur_metric = calculate_blur_metric(img)
        retinal_analysis = analyze_retinal_layers(img)
        
        # Calculate overall quality score
        quality_factors = {
            'snr_score': min(snr / 20.0, 1.0),  # Normalize to 0-1
            'motion_score': 1.0 - motion_artifacts['motion_artifact_score'],
            'contrast_score': min(contrast / 0.5, 1.0),  # Normalize to 0-1
            'saturation_score': 1.0 - max(saturation['over_saturated_ratio'], saturation['under_saturated_ratio']),
            'blur_score': min(blur_metric / 1000.0, 1.0),  # Normalize to 0-1
            'layer_score': min(retinal_analysis['layer_contrast'] / 50.0, 1.0)  # Normalize to 0-1
        }
        
        # Weighted overall score
        weights = {
            'snr_score': 0.25,
            'motion_score': 0.20,
            'contrast_score': 0.20,
            'saturation_score': 0.15,
            'blur_score': 0.10,
            'layer_score': 0.10
        }
        
        overall_score = sum(quality_factors[k] * weights[k] for k in quality_factors.keys())
        
        # Determine if image is processable
        processable = (
            snr > QUALITY_THRESHOLDS['snr_threshold'] and
            not motion_artifacts['has_significant_artifacts'] and
            not saturation['has_saturation_issues'] and
            contrast > QUALITY_THRESHOLDS['contrast_threshold'] and
            blur_metric > QUALITY_THRESHOLDS['blur_threshold']
        )
        
        return {
            'processable': processable,
            'overall_quality_score': overall_score,
            'snr_db': snr,
            'motion_artifacts': motion_artifacts,
            'contrast': contrast,
            'saturation': saturation,
            'blur_metric': blur_metric,
            'retinal_analysis': retinal_analysis,
            'quality_factors': quality_factors,
            'detailed_assessment': {
                'snr_pass': snr > QUALITY_THRESHOLDS['snr_threshold'],
                'motion_pass': not motion_artifacts['has_significant_artifacts'],
                'saturation_pass': not saturation['has_saturation_issues'],
                'contrast_pass': contrast > QUALITY_THRESHOLDS['contrast_threshold'],
                'blur_pass': blur_metric > QUALITY_THRESHOLDS['blur_threshold']
            }
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive quality assessment: {str(e)}")
        return {'error': str(e), 'processable': False}

def assess_image_quality_fast(image_path):
    """Fast quality assessment for basic filtering"""
    try:
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {'error': 'Could not load image', 'processable': False}
        else:
            img = image_path
        
        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Quick checks
        mean = np.mean(img)
        std = np.std(img)
        snr = mean / (std + 1e-6)
        snr_db = 20 * np.log10(snr)
        
        # Motion artifacts (simple version)
        diff = np.diff(img, axis=1)
        motion_score = np.var(diff) / (np.var(img) + 1e-6)
        
        # Saturation
        saturation_high = np.mean(img > 250)
        saturation_low = np.mean(img < 5)
        
        # Basic processability check
        processable = (
            snr_db > QUALITY_THRESHOLDS['snr_threshold'] and
            motion_score < QUALITY_THRESHOLDS['motion_artifact_threshold'] and
            saturation_high < QUALITY_THRESHOLDS['saturation_high_threshold'] and
            saturation_low < QUALITY_THRESHOLDS['saturation_low_threshold']
        )
        
        return {
            'processable': processable,
            'snr_db': snr_db,
            'motion_score': motion_score,
            'saturation_high': saturation_high,
            'saturation_low': saturation_low
        }
        
    except Exception as e:
        logger.error(f"Error in fast quality assessment: {str(e)}")
        return {'error': str(e), 'processable': False}

def generate_quality_report(image_path, output_path=None):
    """Generate detailed quality assessment report with visualizations"""
    try:
        # Get comprehensive assessment
        assessment = assess_image_quality_comprehensive(image_path)
        
        if 'error' in assessment:
            return assessment
        
        # Load original image for visualization
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if isinstance(image_path, str) else image_path
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'OCT Image Quality Assessment Report', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Motion artifacts
        if assessment['motion_artifacts']['artifact_mask'] is not None:
            axes[0, 1].imshow(assessment['motion_artifacts']['artifact_mask'], cmap='hot')
            axes[0, 1].set_title(f'Motion Artifacts\\n(Score: {assessment["motion_artifacts"]["motion_artifact_score"]:.3f})')
        else:
            axes[0, 1].text(0.5, 0.5, 'No artifacts detected', ha='center', va='center')
            axes[0, 1].set_title('Motion Artifacts')
        axes[0, 1].axis('off')
        
        # Intensity profile
        if assessment['retinal_analysis']['intensity_profile'] is not None:
            profile = assessment['retinal_analysis']['intensity_profile']
            axes[0, 2].plot(profile)
            axes[0, 2].set_title('Intensity Profile')
            axes[0, 2].set_xlabel('Depth')
            axes[0, 2].set_ylabel('Intensity')
            
            # Mark detected peaks
            peaks = assessment['retinal_analysis']['detected_peaks']
            if len(peaks) > 0:
                axes[0, 2].plot(peaks, profile[peaks], 'ro', markersize=5)
        
        # Quality metrics bar chart
        metrics = assessment['quality_factors']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in metric_values]
        axes[1, 0].bar(range(len(metric_names)), metric_values, color=colors)
        axes[1, 0].set_xticks(range(len(metric_names)))
        axes[1, 0].set_xticklabels([name.replace('_', '\\n') for name in metric_names], rotation=45, ha='right')
        axes[1, 0].set_title('Quality Metrics')
        axes[1, 0].set_ylabel('Score (0-1)')
        axes[1, 0].set_ylim(0, 1)
        
        # Overall assessment summary
        axes[1, 1].axis('off')
        summary_text = f"""Overall Quality Score: {assessment['overall_quality_score']:.2f}\
"""
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, \
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Assessment Summary')
        
        # Histogram
        axes[1, 2].hist(img.ravel(), bins=50, alpha=0.7, color='blue')
        axes[1, 2].set_title('Intensity Histogram')
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Quality report saved to {output_path}")
        
        return {
            'assessment': assessment,
            'figure': fig,
            'report_path': output_path
        }
        
    except Exception as e:
        logger.error(f"Error generating quality report: {str(e)}")
        return {'error': str(e)}

def batch_quality_assessment(image_paths, output_dir=None):
    """Perform quality assessment on batch of images"""
    try:
        results = []
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            assessment = assess_image_quality_comprehensive(img_path)
            
            result = {
                'filename': filename,
                'path': img_path,
                **assessment
            }
            results.append(result)
        
        # Calculate batch statistics
        processable_count = sum(1 for r in results if r.get('processable', False))
        avg_quality = np.mean([r.get('overall_quality_score', 0) for r in results])
        avg_snr = np.mean([r.get('snr_db', 0) for r in results])
        
        batch_stats = {
            'total_images': len(results),
            'processable_images': processable_count,
            'processable_percentage': (processable_count / len(results)) * 100,
            'average_quality_score': avg_quality,
            'average_snr': avg_snr
        }
        
        # Generate batch visualization if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create batch summary plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Batch Quality Assessment Summary', fontsize=16)
            
            # Quality score distribution
            quality_scores = [r.get('overall_quality_score', 0) for r in results]
            axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Quality Score Distribution')
            axes[0, 0].set_xlabel('Quality Score')
            axes[0, 0].set_ylabel('Frequency')
            
            # SNR distribution
            snr_values = [r.get('snr_db', 0) for r in results]
            axes[0, 1].hist(snr_values, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('SNR Distribution')
            axes[0, 1].set_xlabel('SNR (dB)')
            axes[0, 1].set_ylabel('Frequency')
            
            # Processable vs non-processable
            processable_counts = [processable_count, len(results) - processable_count]
            axes[1, 0].pie(processable_counts, labels=['Processable', 'Non-processable'], \
                          autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Image Processability')
            
            # Quality factors comparison
            if results and 'quality_factors' in results[0]:
                factor_names = list(results[0]['quality_factors'].keys())
                factor_averages = [np.mean([r['quality_factors'].get(factor, 0) for r in results]) \
                                 for factor in factor_names]
                
                axes[1, 1].bar(range(len(factor_names)), factor_averages)
                axes[1, 1].set_xticks(range(len(factor_names)))
                axes[1, 1].set_xticklabels([name.replace('_', '\\n') for name in factor_names], \
                                          rotation=45, ha='right')
                axes[1, 1].set_title('Average Quality Factors')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            summary_path = os.path.join(output_dir, 'batch_quality_summary.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save detailed results as CSV
            import pandas as pd
            df_data = []
            for r in results:
                row = {
                    'filename': r['filename'],
                    'processable': r.get('processable', False),
                    'overall_quality_score': r.get('overall_quality_score', 0),
                    'snr_db': r.get('snr_db', 0),
                    'motion_score': r.get('motion_artifacts', {}).get('motion_artifact_score', 0),
                    'contrast': r.get('contrast', 0),
                    'blur_metric': r.get('blur_metric', 0)
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(output_dir, 'batch_quality_results.csv')
            df.to_csv(csv_path, index=False)
            
            batch_stats['summary_plot'] = summary_path
            batch_stats['detailed_csv'] = csv_path
        
        return {
            'results': results,
            'batch_statistics': batch_stats
        }
        
    except Exception as e:
        logger.error(f"Error in batch quality assessment: {str(e)}")
        return {'error': str(e)}

def get_quality_recommendations(assessment):
    """Get recommendations for improving image quality"""
    recommendations = []
    
    if not assessment.get('processable', True):
        detailed = assessment.get('detailed_assessment', {})
        
        if not detailed.get('snr_pass', True):
            recommendations.append("Improve signal-to-noise ratio by adjusting acquisition parameters or reducing system noise")
        
        if not detailed.get('motion_pass', True):
            recommendations.append("Reduce motion artifacts by ensuring patient fixation and minimizing head movement")
        
        if not detailed.get('contrast_pass', True):
            recommendations.append("Improve image contrast by adjusting illumination or gain settings")
        
        if not detailed.get('saturation_pass', True):
            recommendations.append("Adjust exposure settings to avoid over/under-saturation")
        
        if not detailed.get('blur_pass', True):
            recommendations.append("Improve focus and reduce blur by proper alignment and focus adjustment")
    
    if not recommendations:
        recommendations.append("Image quality is acceptable for analysis")
    
    return recommendations