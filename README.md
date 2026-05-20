# Deep-Retina OCT Image Analysis

A comprehensive web application for analyzing Optical Coherence Tomography (OCT) retinal images using deep learning models. Built with Flask, PyTorch, and advanced computer vision techniques for medical image analysis.

## 🎯 Overview

This application provides an interactive platform for:
- **OCT Image Upload & Analysis** - Upload retinal OCT images for automated analysis
- **Multiple ML Models** - Support for various deep learning models for image classification
- **Image Quality Assessment** - Comprehensive image quality evaluation with recommendations
- **Visualization Suite** - Advanced visualizations and heatmaps for interpretability
- **Performance Benchmarking** - Compare model performance with detailed metrics
- **Export & Reporting** - Generate downloadable analysis reports and results

## 🌟 Features

- 🔬 **Medical Image Analysis** - Specialized for OCT retinal imaging
- 🤖 **Multiple AI Models** - Support for various pre-trained deep learning models
- 📊 **Advanced Visualizations** - Heatmaps, confusion matrices, and detailed analytics
- 🎨 **Interactive Web Interface** - User-friendly Flask-based UI
- ✅ **Quality Assessment** - Automatic image quality validation and recommendations
- 📈 **Benchmarking Tools** - Compare model performance and accuracy metrics
- 🐳 **Docker Support** - Easy containerization and deployment
- 🔒 **CSRF Protection** - Secure web forms with Flask-WTF
- 📁 **Batch Processing** - Analyze multiple images at once
- 💾 **Export Results** - Download analysis reports and visualizations

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask 2.3.3
- **Server**: Gunicorn 21.2.0
- **Python**: 3.11+

### Machine Learning & Computer Vision
- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV, scikit-image, Pillow
- **ML Utilities**: scikit-learn
- **Data Processing**: NumPy, SciPy, Pandas

### Visualization
- **Plotting**: Matplotlib, Seaborn
- **Interactive Charts**: Chart.js

### Frontend
- **Template Engine**: Jinja2
- **Styling**: HTML5/CSS3
- **Interactivity**: Vanilla JavaScript

### DevOps
- **Containerization**: Docker
- **Web Server**: Nginx (reverse proxy)

## 📋 Requirements

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- Docker & Docker Compose (for containerized deployment)
- 4GB+ RAM
- GPU recommended for faster inference

## 🚀 Getting Started

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Deep-Retina-OCT-image-analysis.git
cd Deep-Retina-OCT-image-analysis
```

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and visit:
```
http://localhost:5000
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -f Dockerfile -t oct-analysis:latest .
```

2. Run the container:
```bash
docker run -p 5000:5000 oct-analysis:latest
```

3. Access the application at `http://localhost:5000`

### Docker Compose Deployment

```bash
docker-compose up -d
```

The application will be available at `http://localhost` (via Nginx reverse proxy)

## 📁 Project Structure

```
Deep-Retina-OCT-image-analysis/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── nginx.conf                  # Nginx reverse proxy config
├── templates/                  # HTML templates
│   └── index.html             # Main web interface
├── static/                     # Static files
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript files
│   └── images/                # Images and assets
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py       # Image preprocessing
│   ├── model_loader.py        # Model loading and management
│   ├── visualization.py       # Visualization generation
│   ├── quality_assessment.py  # Image quality metrics
│   ├── benchmarking.py        # Performance benchmarking
│   └── advanced_visualization.py  # Advanced visualization suite
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:
```bash
FLASK_ENV=production
FLASK_DEBUG=0
PYTHONUNBUFFERED=1
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=52428800  # 50MB max file size
```

### Model Configuration

Models and metadata are loaded from `model_loader.py`. Add custom models by:
1. Adding model files to the models directory
2. Registering in `get_all_model_metadata()` function
3. Updating preprocessing pipelines if needed

## 📊 Usage Guide

### Analyzing a Single Image

1. Open the web application
2. Click "Upload OCT Image"
3. Select an OCT image file (PNG, JPG, DICOM format)
4. Choose an AI model for analysis
5. Click "Analyze"
6. View results, visualizations, and quality metrics

### Batch Analysis

1. Upload multiple images
2. Select model and processing options
3. Start batch analysis
4. Download combined report when complete

### Quality Assessment

The application automatically:
- Checks image dimensions
- Validates pixel intensity ranges
- Assesses signal-to-noise ratio
- Provides improvement recommendations

### Benchmarking

Compare model performance:
1. Select multiple models
2. Run on test dataset
3. View confusion matrices and metrics
4. Export comparison report

## 🤖 Supported Models

The application supports various deep learning models including:
- ResNet-based architectures
- Vision Transformers
- Custom medical imaging models
- Ensemble models

Add new models by modifying `utils/model_loader.py`

## 📊 Visualization Features

- **Classification Results** - Class predictions with confidence scores
- **Heatmaps** - Attention/activation maps showing analysis regions
- **Confusion Matrix** - Model performance visualization
- **Quality Metrics** - Image quality visualization
- **ROC Curves** - Model performance curves
- **Statistical Reports** - Detailed analysis statistics

## 🔒 Security Features

- CSRF protection on all forms
- Secure file upload handling
- Input validation
- Error handling and logging
- Rate limiting support

## 📈 Performance

### Optimization

- GPU acceleration support
- Image preprocessing optimization
- Batch processing capability
- Caching mechanisms
- Efficient model inference

### Benchmarks

- Average inference time: 100-500ms per image (depends on model)
- Quality assessment: < 50ms per image
- Visualization generation: 200-1000ms

## 🐛 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Model loading errors:**
- Ensure all model files are in correct directory
- Check model compatibility with installed PyTorch version

**High memory usage:**
- Reduce batch size
- Enable model quantization
- Use GPU acceleration

### Logging

Check application logs for debugging:
```bash
# View logs
tail -f app.log
```

## 🚀 Deployment

### Production Deployment

1. **AWS EC2/ECS:**
   - Push Docker image to ECR
   - Deploy using ECS or Fargate
   - Configure load balancer

2. **Heroku:**
   ```bash
   git push heroku main
   ```

3. **DigitalOcean:**
   - Use App Platform
   - Select Docker as deployment method

4. **GCP/Azure:**
   - Use Cloud Run or App Service
   - Configure environment variables

### Performance Tuning

- Use production WSGI server (Gunicorn)
- Configure Nginx caching
- Enable GZIP compression
- Optimize image preprocessing

## 📝 API Endpoints

- `GET /` - Main application interface
- `POST /analyze` - Analyze single image
- `POST /batch-analyze` - Batch analysis
- `GET /results/<id>` - Retrieve analysis results
- `GET /benchmark` - Run model benchmarking
- `GET /quality-check/<id>` - Image quality assessment

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ML models
- Performance optimization
- UI/UX enhancements
- New visualization types
- Documentation improvements

## 📜 License

MIT License - Copyright (c) 2025 Abhijith Krishna G

See [LICENSE](LICENSE) file for details.

## 📧 Contact

- Email: abhijithkrishnag234@gmail.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Medical imaging datasets and validation
- Open-source ML frameworks (PyTorch, scikit-learn)
- Computer vision libraries (OpenCV)

## ⚖️ Medical Disclaimer

This application is for research and educational purposes only. It is not intended for clinical diagnosis. Always consult qualified medical professionals for medical decisions.

---

Made with ❤️ by Abhijith Krishna G
