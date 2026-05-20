#!/usr/bin/env python3
"""
Test script to verify webapp dependencies and models
"""

import sys
import os
import importlib

def test_import(module_name, description=""):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False

def test_models():
    """Test if model files exist"""
    model_paths = {
        'CNN Model': 'models/oct_model_pytorch.pth',
        'Swin Model': 'models/swin_model.pth',
        'ViT Model': 'models/vit_oct_model/model.safetensors'
    }
    
    print("\n🔍 Checking Model Files:")
    for name, path in model_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")

def test_webapp_imports():
    """Test webapp-specific imports"""
    print("\n🧪 Testing Webapp Imports:")
    
    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())
    
    try:
        from utils.model_loader import load_model, get_all_model_metadata
        print("✅ utils.model_loader")
        
        from utils.preprocessing import preprocess_image
        print("✅ utils.preprocessing")
        
        from utils.visualization import create_visualization
        print("✅ utils.visualization")
        
        from utils.quality_assessment import assess_image_quality_comprehensive
        print("✅ utils.quality_assessment")
        
    except ImportError as e:
        print(f"❌ Webapp utils import failed: {e}")

def main():
    print("🚀 OCT Webapp Dependency Test")
    print("=" * 50)
    
    # Test core dependencies
    print("\n📦 Testing Core Dependencies:")
    all_good = True
    
    dependencies = [
        ("flask", "Web framework"),
        ("torch", "PyTorch"),
        ("torchvision", "PyTorch vision"),
        ("transformers", "Hugging Face transformers"),
        ("timm", "PyTorch image models"),
        ("cv2", "OpenCV"),
        ("numpy", "Numerical computing"),
        ("matplotlib", "Plotting"),
        ("pandas", "Data manipulation"),
        ("seaborn", "Statistical visualization"),
        ("sklearn", "Machine learning"),
        ("PIL", "Image processing")
    ]
    
    for module, desc in dependencies:
        if not test_import(module, desc):
            all_good = False
    
    # Test model files
    test_models()
    
    # Test webapp imports
    test_webapp_imports()
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All dependencies look good! Ready to run the webapp.")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
    else:
        print("⚠️  Some dependencies are missing. Check the errors above.")
        print("\nTry:")
        print("1. pip install -r requirements.txt")
        print("2. Or: pip install -r requirements-updated.txt")

if __name__ == "__main__":
    main()
