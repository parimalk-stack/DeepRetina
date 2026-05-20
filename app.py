"""OCT Image Analysis Web Application.

This module provides a web interface for analyzing OCT images using various deep learning models.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session, send_file
from flask_wtf.csrf import CSRFProtect
import os
import cv2
import numpy as np
import torch
import sys
import time
import uuid
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import io
import zipfile
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import utility functions
from utils.preprocessing import preprocess_image
from utils.model_loader import load_model, get_all_model_metadata, preprocess_for_model
from utils.visualization import create_visualization
from utils.quality_assessment import assess_image_quality_comprehensive, assess_image_quality_fast, get_quality_recommendations
from utils.benchmarking import create_model_benchmark
from utils.advanced_visualization import create_visualization_suite

# Create Flask app with explicit static folder
app = Flask(__name__, 
           static_folder='static',
           static_url_path='/static')

# Configure CSRF protection
app.config['SECRET_KEY'] = os.urandom(24)  # Generate a random secret key
csrf = CSRFProtect(app)

# Configure upload settings
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_EXTENSIONS'] = ['.png', '.jpg', '.jpeg']
app.secret_key = 'oct-analysis-secret-key-change-in-production'  # For session management

# Ensure upload directory exists with proper permissions
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define model paths
MODEL_PATHS = {
    'cnn_model': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'oct_model_pytorch.pth'),
    'vit_model': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'vit_oct_model'),
    'swin_model': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'swin_model.pth')
}

# Load available models
AVAILABLE_MODELS = {}
try:
    logger.info("Loading models...")
    
    # Load CNN model
    try:
        AVAILABLE_MODELS['cnn_model'] = load_model(MODEL_PATHS['cnn_model'], 'cnn_model')
        logger.info("CNN model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CNN model: {str(e)}")
    
    # Load ViT model
    try:
        AVAILABLE_MODELS['vit_model'] = load_model(MODEL_PATHS['vit_model'], 'vit_model')
        logger.info("ViT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ViT model: {str(e)}")
    
    # Load Swin model
    try:
        AVAILABLE_MODELS['swin_model'] = load_model(MODEL_PATHS['swin_model'], 'swin_model')
        logger.info("Swin model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Swin model: {str(e)}")
    
    logger.info(f"Successfully loaded {len(AVAILABLE_MODELS)} models: {list(AVAILABLE_MODELS.keys())}")
    
except Exception as e:
    logger.error(f"Error during model loading: {str(e)}")
    AVAILABLE_MODELS = {}

# Get model metadata
MODEL_METADATA = get_all_model_metadata()

# Define constants
IMG_SIZE = 224  # Must match the size used during training
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class mapping 
CLASS_MAPPING = {
    0: 'CNV',
    1: 'DME',
    2: 'DRUSEN',
    3: 'NORMAL'
}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page with upload form"""
    # Add model metadata to the template context
    return render_template('index.html', 
                         models=list(AVAILABLE_MODELS.keys()),
                         model_metadata=MODEL_METADATA)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle single file upload and analysis"""
    logger.info("Processing upload request")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename to prevent conflicts
            unique_id = uuid.uuid4().hex
            filename = f"{unique_id}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to {filepath}")
            
            # Ensure the upload directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the file
            file.save(filepath)
            
            # Get selected model 
            model_name = request.form.get('model', 'cnn_model')
            logger.info(f"Using model: {model_name}")
            
            # Process the image
            result = process_image(filepath, model_name)
            
            # Construct URLs for images
            original_url = f"/static/uploads/{filename}"
            
            logger.info(f"Returning result with image URL: {original_url}")
            return jsonify({
                'result': result,
                'filename': filename
            })
        except Exception as e:
            logger.error(f"Error in upload: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)})
    
    logger.warning(f"Invalid file format: {file.filename}")
    return jsonify({'error': 'Invalid file format'})

@app.route('/compare', methods=['POST'])
def compare_models():
    """Compare multiple models on the same image"""
    logger.info("Processing comparison request")
    
    # Debug: Print all form data
    logger.info(f"Request form data: {dict(request.form)}")
    logger.info(f"Request files: {list(request.files.keys())}")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            unique_id = uuid.uuid4().hex
            filename = f"{unique_id}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to {filepath}")
            file.save(filepath)
            
            # Get selected models
            model_names = request.form.getlist('models')
            logger.info(f"Raw model_names from form: {model_names}")
            logger.info(f"Available models in AVAILABLE_MODELS: {list(AVAILABLE_MODELS.keys())}")
            
            if not model_names:
                model_names = ['cnn_model']  # Default
                logger.warning("No models selected, using default cnn_model")
            
            logger.info(f"Final models to compare: {', '.join(model_names)}")
            
            # Compare models
            results = {}
            successful_models = []
            
            for model_name in model_names:
                if model_name in AVAILABLE_MODELS:
                    logger.info(f"Processing with model: {model_name}")
                    try:
                        result = process_image(filepath, model_name)
                        if 'error' not in result:
                            results[model_name] = result
                            successful_models.append(model_name)
                            logger.info(f"Successfully processed with {model_name}")
                        else:
                            logger.error(f"Error processing with {model_name}: {result['error']}")
                    except Exception as e:
                        logger.error(f"Exception processing with {model_name}: {str(e)}")
                else:
                    logger.warning(f"Model {model_name} not found in AVAILABLE_MODELS")
            
            logger.info(f"Successfully processed {len(results)} models: {list(results.keys())}")
            
            return jsonify({
                'results': results,
                'filename': filename,
                'debug_info': {
                    'requested_models': model_names,
                    'successful_models': successful_models,
                    'available_models': list(AVAILABLE_MODELS.keys())
                }
            })
        except Exception as e:
            logger.error(f"Error in comparison: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)})
    
    logger.warning(f"Invalid file format: {file.filename}")
    return jsonify({'error': 'Invalid file format'})

def process_image(filepath, model_name):
    """Process an image with the specified model"""
    try:
        logger.info(f"Processing image: {filepath} with model: {model_name}")
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            logger.error(f"Failed to load image: {filepath}")
            return {"error": f"Failed to load image: File may be corrupted or in an unsupported format"}
        
        logger.info(f"Image shape: {img.shape}")
        
        # Preprocess image
        start_time = time.time()
        processed_img, processed_img_path = preprocess_image(img, app.config['UPLOAD_FOLDER'])
        processing_time = time.time() - start_time
        
        # Get model
        model = AVAILABLE_MODELS.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return {'error': f'Model {model_name} not found'}
        
        # Prepare input for specific model type
        start_time = time.time()
        try:
            if model_name == 'cnn_model':
                # CNN expects grayscale (batch_size, channels, height, width)
                # Normalize the processed image to [0, 1] range
                normalized_img = processed_img.astype(np.float32) / 255.0
                # Add batch and channel dimensions: (224, 224) -> (1, 1, 224, 224)
                input_tensor = torch.FloatTensor(normalized_img).unsqueeze(0).unsqueeze(0).to('cpu')
            elif model_name in ['vit_model', 'swin_model']:
                # ViT and Swin expect different preprocessing
                input_data = preprocess_for_model(processed_img, model_name)
            
            # Get prediction
            with torch.no_grad():
                if model_name == 'cnn_model':
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                elif model_name in ['vit_model', 'swin_model']:
                    outputs = model(input_data)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    if outputs.dim() == 2:
                        outputs = outputs.squeeze(0)
                    probabilities = torch.nn.functional.softmax(outputs, dim=0)
                
                predicted_class = torch.argmax(probabilities).item()
                
        except Exception as e:
            logger.error(f"Error during model inference: {str(e)}")
            return {'error': f'Model inference failed: {str(e)}'}
            
        inference_time = time.time() - start_time
        
        # Get class probabilities
        probs = probabilities.cpu().numpy() if hasattr(probabilities, 'cpu') else probabilities
        class_probs = {CLASS_MAPPING[i]: float(probs[i]) for i in range(len(probs))}
        
        # Read the 2D processed image for visualization
        processed_img_2d = cv2.imread(processed_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Create visualization
        viz_filename = f"viz_{model_name}_{os.path.basename(filepath)}"
        viz_path = os.path.join(app.config['UPLOAD_FOLDER'], viz_filename)
        
        create_visualization(img, processed_img_2d, CLASS_MAPPING[predicted_class], 
                           class_probs, viz_path)
        
        # Extract true class from filename if available
        true_class = None
        filename = os.path.basename(filepath)
        for class_name in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
            if class_name.lower() in filename.lower():
                true_class = class_name
                break
        
        # Prepare result
        result = {
            'predicted_class': CLASS_MAPPING[predicted_class],
            'confidence': float(probs[predicted_class]),
            'all_probabilities': class_probs,
            'processing_time': processing_time,
            'inference_time': inference_time,
            'visualization': f"uploads/{viz_filename}",
            'filename': filename,
            'true_class': true_class,
            'correct': CLASS_MAPPING[predicted_class] == true_class if true_class else None,
            'timestamp': datetime.now().isoformat(),
            'model_used': model_name,
            'model_metadata': MODEL_METADATA.get(model_name, {})
        }
        
        # Store in session history
        if 'analysis_history' not in session:
            session['analysis_history'] = []
        
        session['analysis_history'].append(result)
        
        # Keep only last 50 analyses in session
        if len(session['analysis_history']) > 50:
            session['analysis_history'] = session['analysis_history'][-50:]
        
        session.modified = True
        
        logger.info(f"Prediction: {result['predicted_class']} with confidence {result['confidence']}")
        return result
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return {"error": str(e)}

# New routes for enhanced functionality

@app.route('/quality_assessment', methods=['POST'])
def quality_assessment():
    """Perform quality assessment on uploaded image"""
    logger.info("Processing quality assessment request")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            unique_id = uuid.uuid4().hex
            filename = f"{unique_id}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to {filepath}")
            file.save(filepath)
            
            # Perform comprehensive quality assessment
            quality_result = assess_image_quality_comprehensive(filepath)
            
            # Add recommendations
            recommendations = get_quality_recommendations(quality_result)
            
            # Generate quality visualization
            from utils.advanced_visualization import OCTVisualization
            viz = OCTVisualization()
            
            viz_filename = f"quality_viz_{filename}"
            viz_path = os.path.join(app.config['UPLOAD_FOLDER'], viz_filename)
            
            quality_fig = viz.create_quality_assessment_visualization(quality_result, viz_path)
            
            # Format response to match expected format
            response = {
                'filename': filename,
                'visualization': f"uploads/{viz_filename}",
                'quality_score': quality_result.get('overall_quality_score', 0),
                'snr': quality_result.get('snr_db', 0),
                'motion_artifacts': quality_result.get('motion_artifacts', {}).get('score', 0),
                'contrast': quality_result.get('contrast', 0),
                'blur_metric': quality_result.get('blur_metric', 0),
                'status': "Good" if quality_result.get('processable', False) else "Poor",
                'recommendations': recommendations
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)})
    
    logger.warning(f"Invalid file format: {file.filename}")
    return jsonify({'error': 'Invalid file format'})

@app.route('/enhanced_compare', methods=['POST'])
def enhanced_compare():
    """Enhanced model comparison with benchmarking"""
    logger.info("Processing enhanced comparison request")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            unique_id = uuid.uuid4().hex
            filename = f"{unique_id}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to {filepath}")
            file.save(filepath)
            
            # Get selected models
            model_names = request.form.getlist('models')
            if not model_names:
                model_names = list(AVAILABLE_MODELS.keys())  # Use all available models
            
            logger.info(f"Comparing models: {', '.join(model_names)}")
            
            # Perform quality assessment first
            quality_result = assess_image_quality_fast(filepath)
            
            # Compare models
            results = {}
            benchmark = create_model_benchmark()
            
            for model_name in model_names:
                if model_name in AVAILABLE_MODELS:
                    logger.info(f"Processing with model: {model_name}")
                    result = process_image(filepath, model_name)
                    
                    if 'error' not in result:
                        results[model_name] = result
                        benchmark.add_result(model_name, result)
                        benchmark.set_model_metadata(model_name, MODEL_METADATA.get(model_name, {}))
            
            # Generate comprehensive comparison
            comparison_analysis = benchmark.compare_models(list(results.keys()))
            
            # Create enhanced visualizations
            viz_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"comparison_{unique_id}")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Generate comparison charts
            from utils.advanced_visualization import OCTVisualization
            viz = OCTVisualization()
            
            comparison_chart = viz.create_model_comparison_chart(results)
            if comparison_chart:
                chart_path = os.path.join(viz_dir, 'comparison_chart.html')
                comparison_chart.write_html(chart_path)
            
            return jsonify({
                'results': results,
                'comparison_analysis': comparison_analysis,
                'quality_assessment': quality_result,
                'visualizations': {
                    'comparison_chart': f"uploads/comparison_{unique_id}/comparison_chart.html"
                },
                'filename': filename,
                'model_metadata': {model: MODEL_METADATA.get(model, {}) for model in results.keys()}
            })
            
        except Exception as e:
            logger.error(f"Error in enhanced comparison: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)})
    
    logger.warning(f"Invalid file format: {file.filename}")
    return jsonify({'error': 'Invalid file format'})

@app.route('/batch_quality_assessment', methods=['POST'])
def batch_quality_assessment():
    """Perform batch quality assessment"""
    logger.info("Processing batch quality assessment request")
    
    if 'files[]' not in request.files:
        logger.warning("No files in batch request")
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        logger.warning("No selected files in batch")
        return jsonify({'error': 'No selected files'})
    
    results = []
    processed_files = 0
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                # Generate unique filename
                unique_id = uuid.uuid4().hex
                filename = f"{unique_id}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(filepath)
                    logger.info(f"Processing quality assessment for: {file.filename}")
                    
                    # Perform quality assessment
                    quality_result = assess_image_quality_comprehensive(filepath)
                    quality_result['original_filename'] = file.filename
                    quality_result['saved_filename'] = filename
                    
                    results.append(quality_result)
                    processed_files += 1
                    
                    # Clean up temporary file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        
                except Exception as e:
                    logger.error(f"Error processing {file.filename}: {str(e)}")
                    continue
        
        if not results:
            return jsonify({'error': 'No files were successfully processed'})
        
        # Calculate batch statistics
        total_processable = sum(1 for r in results if r.get('processable', False))
        avg_quality_score = np.mean([r.get('overall_quality_score', 0) for r in results])
        avg_snr = np.mean([r.get('snr_db', 0) for r in results])
        
        batch_stats = {
            'total_files': len(results),
            'processable_files': total_processable,
            'processable_percentage': (total_processable / len(results)) * 100,
            'average_quality_score': avg_quality_score,
            'average_snr': avg_snr
        }
        
        logger.info(f"Batch quality assessment completed: {processed_files} files processed")
        
        return jsonify({
            'results': results,
            'batch_statistics': batch_stats,
            'total_processed': processed_files
        })
        
    except Exception as e:
        logger.error(f"Error in batch quality assessment: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Handle batch file upload and analysis"""
    logger.info("Processing batch analysis request")
    
    if 'files[]' not in request.files:
        logger.warning("No files in batch request")
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        logger.warning("No selected files in batch")
        return jsonify({'error': 'No selected files'})
    
    # Get selected model
    model_name = request.form.get('model', 'cnn_model')
    logger.info(f"Batch processing with model: {model_name}")
    
    results = []
    processed_files = 0
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                # Generate unique filename
                unique_id = uuid.uuid4().hex
                filename = f"{unique_id}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(filepath)
                    logger.info(f"Processing batch file: {file.filename}")
                    
                    # Process the image
                    result = process_image(filepath, model_name)
                    
                    if 'error' not in result:
                        results.append(result)
                        processed_files += 1
                    
                    # Clean up temporary file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        
                except Exception as e:
                    logger.error(f"Error processing {file.filename}: {str(e)}")
                    continue
        
        if not results:
            return jsonify({'error': 'No files were successfully processed'})
        
        # Calculate batch metrics
        batch_metrics = calculate_batch_metrics(results)
        
        # Generate batch visualizations
        batch_viz = generate_batch_visualizations(results)
        
        logger.info(f"Batch processing completed: {processed_files} files processed")
        
        return jsonify({
            'results': results,
            'metrics': batch_metrics,
            'visualizations': batch_viz,
            'total_processed': processed_files
        })
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

@app.route('/get_history')
def get_history():
    """Get user's analysis history"""
    history = session.get('analysis_history', [])
    
    # Calculate session metrics
    session_metrics = calculate_session_metrics(history)
    
    return jsonify({
        'history': history,
        'metrics': session_metrics
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear user's analysis history"""
    session['analysis_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/export_results', methods=['POST'])
def export_results():
    """Export analysis results to CSV"""
    try:
        export_type = request.json.get('type', 'session')  # 'session' or 'batch'
        
        if export_type == 'session':
            data = session.get('analysis_history', [])
        else:
            # For batch export, get data from request
            data = request.json.get('data', [])
        
        if not data:
            return jsonify({'error': 'No data to export'})
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Select relevant columns
        export_columns = [
            'filename', 'predicted_class', 'true_class', 'correct', 
            'confidence', 'processing_time', 'inference_time', 'timestamp'
        ]
        
        # Add probability columns
        for class_name in CLASS_MAPPING.values():
            prob_col = f'{class_name.lower()}_probability'
            if 'all_probabilities' in df.columns:
                df[prob_col] = df['all_probabilities'].apply(
                    lambda x: x.get(class_name, 0) if isinstance(x, dict) else 0
                )
                export_columns.append(prob_col)
        
        # Filter columns that exist
        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns]
        
        # Create CSV in memory
        output = io.StringIO()
        export_df.to_csv(output, index=False)
        
        # Create response
        response_data = {
            'csv_data': output.getvalue(),
            'filename': f'oct_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

@app.route('/export_report', methods=['POST'])
def export_report():
    """Export comprehensive HTML report"""
    try:
        data = request.json.get('data', [])
        if not data:
            data = session.get('analysis_history', [])
        
        if not data:
            return jsonify({'error': 'No data to export'})
        
        # Generate comprehensive report
        report_html = generate_comprehensive_report(data)
        
        response_data = {
            'html_data': report_html,
            'filename': f'oct_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

# Helper functions

def calculate_batch_metrics(results):
    """Calculate metrics for batch processing"""
    if not results:
        return {}
    
    # Basic metrics
    total_files = len(results)
    
    # Confidence statistics
    confidences = [r['confidence'] for r in results]
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    
    # Class distribution
    class_distribution = Counter([r['predicted_class'] for r in results])
    
    # Accuracy metrics (if ground truth available)
    accuracy_metrics = {}
    results_with_truth = [r for r in results if r.get('true_class')]
    
    if results_with_truth:
        correct_predictions = sum(1 for r in results_with_truth if r.get('correct'))
        total_with_truth = len(results_with_truth)
        overall_accuracy = correct_predictions / total_with_truth
        
        # Per-class accuracy
        class_accuracy = {}
        for class_name in CLASS_MAPPING.values():
            class_results = [r for r in results_with_truth if r['true_class'] == class_name]
            if class_results:
                class_correct = sum(1 for r in class_results if r['correct'])
                class_accuracy[class_name] = class_correct / len(class_results)
        
        accuracy_metrics = {
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct_predictions,
            'total_with_ground_truth': total_with_truth,
            'class_accuracy': class_accuracy
        }
    
    # Processing time statistics
    processing_times = [r.get('processing_time', 0) + r.get('inference_time', 0) for r in results]
    total_processing_time = np.sum(processing_times)
    avg_processing_time = np.mean(processing_times)
    
    return {
        'total_files': total_files,
        'average_confidence': float(avg_confidence),
        'min_confidence': float(min_confidence),
        'max_confidence': float(max_confidence),
        'class_distribution': dict(class_distribution),
        'total_processing_time': float(total_processing_time),
        'average_processing_time': float(avg_processing_time),
        **accuracy_metrics
    }

def calculate_session_metrics(history):
    """Calculate metrics for session history"""
    if not history:
        return {
            'total_analyses': 0,
            'avg_confidence': 0,
            'class_distribution': {},
            'recent_accuracy': None
        }
    
    # Basic metrics
    total_analyses = len(history)
    
    # Confidence statistics
    confidences = [h['confidence'] for h in history]
    avg_confidence = np.mean(confidences) if confidences else 0
    
    # Class distribution
    class_distribution = Counter([h['predicted_class'] for h in history])
    
    # Recent accuracy (if available)
    recent_accuracy = None
    recent_with_truth = [h for h in history[-20:] if h.get('true_class')]  # Last 20 with ground truth
    if recent_with_truth:
        correct = sum(1 for h in recent_with_truth if h.get('correct'))
        recent_accuracy = correct / len(recent_with_truth)
    
    return {
        'total_analyses': total_analyses,
        'avg_confidence': float(avg_confidence),
        'class_distribution': dict(class_distribution),
        'recent_accuracy': float(recent_accuracy) if recent_accuracy else None
    }

def generate_batch_visualizations(results):
    """Generate visualizations for batch results"""
    try:
        if not results:
            return {}
        
        visualizations = {}
        
        # 1. Class distribution pie chart
        class_dist = Counter([r['predicted_class'] for r in results])
        
        plt.figure(figsize=(8, 6))
        plt.pie(class_dist.values(), labels=class_dist.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution')
        
        class_dist_path = os.path.join(app.config['UPLOAD_FOLDER'], f'class_distribution_{int(time.time())}.png')
        plt.savefig(class_dist_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        visualizations['Class Distribution'] = f"uploads/{os.path.basename(class_dist_path)}"
        
        # 2. Confidence distribution histogram
        confidences = [r['confidence'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.grid(True, alpha=0.3)
        
        conf_dist_path = os.path.join(app.config['UPLOAD_FOLDER'], f'confidence_distribution_{int(time.time())}.png')
        plt.savefig(conf_dist_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        visualizations['Confidence Distribution'] = f"uploads/{os.path.basename(conf_dist_path)}"
        
        # 3. Confusion matrix (if ground truth available)
        results_with_truth = [r for r in results if r.get('true_class')]
        if len(results_with_truth) > 1:
            y_true = [r['true_class'] for r in results_with_truth]
            y_pred = [r['predicted_class'] for r in results_with_truth]
            
            # Create confusion matrix
            classes = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            cm_path = os.path.join(app.config['UPLOAD_FOLDER'], f'confusion_matrix_{int(time.time())}.png')
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualizations['Confusion Matrix'] = f"uploads/{os.path.basename(cm_path)}"
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error generating batch visualizations: {str(e)}", exc_info=True)
        return {}

def generate_comprehensive_report(data):
    """Generate comprehensive HTML report"""
    if not data:
        return "<html><body><h1>No data available for report</h1></body></html>"
    
    # Calculate metrics
    metrics = calculate_batch_metrics(data)
    
    # Generate report HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCT Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e8f4fd; border-radius: 5px; }}
            .correct {{ color: green; font-weight: bold; }}
            .incorrect {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>OCT Image Analysis Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Total Images Analyzed:</strong> {metrics.get('total_files', 0)}</p>
            <p><strong>Average Confidence:</strong> {metrics.get('avg_confidence', 0):.2%}</p>
            <p><strong>Average Processing Time:</strong> {metrics.get('avg_processing_time', 0):.3f} seconds</p>
            {f'<p><strong>Overall Accuracy:</strong> {metrics.get("overall_accuracy", 0):.2%}</p>' if metrics.get('overall_accuracy') else ''}
        </div>
        
        <h2>Class Distribution</h2>
        <table>
            <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
    """
    
    # Add class distribution
    class_dist = metrics.get('class_distribution', {})
    total = sum(class_dist.values()) if class_dist else 1
    
    for class_name, count in class_dist.items():
        percentage = (count / total) * 100
        html_content += f"""
            <tr>
                <td>{class_name}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Individual Results</h2>
        <table>
            <tr>
                <th>Filename</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>True Class</th>
                <th>Result</th>
                <th>Timestamp</th>
            </tr>
    """
    
    # Add individual results
    for result in data:
        correct_class = "correct" if result.get('correct') else "incorrect" if result.get('correct') is False else ""
        result_text = "✓" if result.get('correct') else "✗" if result.get('correct') is False else "N/A"
        
        html_content += f"""
            <tr>
                <td>{result.get('filename', 'N/A')}</td>
                <td>{result.get('predicted_class', 'N/A')}</td>
                <td>{result.get('confidence', 0):.2%}</td>
                <td>{result.get('true_class', 'N/A')}</td>
                <td class="{correct_class}">{result_text}</td>
                <td>{result.get('timestamp', 'N/A')[:19] if result.get('timestamp') else 'N/A'}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    return html_content

if __name__ == '__main__':
    logger.info("Starting OCT Image Analysis Web Application")
    app.run(debug=True, host='0.0.0.0', port=5000)
