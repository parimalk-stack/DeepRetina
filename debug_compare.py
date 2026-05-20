#!/usr/bin/env python3
"""
Debug script to test the Compare Models functionality
Run this after starting your webapp to test the fixes
"""

import requests
import json

def test_compare_models():
    """Test the compare models endpoint"""
    
    print("🔍 Testing Compare Models Functionality")
    print("=" * 50)
    
    # Test URL (make sure your webapp is running on localhost:5000)
    url = "http://localhost:5000/compare"
    
    # Test data - you'll need to provide a real OCT image file
    test_image_path = input("Enter path to an OCT image file for testing: ").strip()
    
    if not test_image_path:
        print("❌ No image path provided")
        return
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'models': ['cnn_model', 'vit_model', 'swin_model'],  # All three models
                'csrf_token': 'test'  # You might need to get the real CSRF token
            }
            
            print(f"🚀 Sending request with models: {data['models']}")
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'error' in result:
                    print(f"❌ Server error: {result['error']}")
                else:
                    print("✅ Request successful!")
                    print(f"📊 Models processed: {list(result.get('results', {}).keys())}")
                    
                    if 'debug_info' in result:
                        debug = result['debug_info']
                        print(f"🔧 Debug info:")
                        print(f"   Requested: {debug.get('requested_models', [])}")
                        print(f"   Successful: {debug.get('successful_models', [])}")
                        print(f"   Available: {debug.get('available_models', [])}")
                    
                    # Check results
                    results = result.get('results', {})
                    for model_name, model_result in results.items():
                        if 'error' in model_result:
                            print(f"❌ {model_name}: {model_result['error']}")
                        else:
                            print(f"✅ {model_name}: {model_result['predicted_class']} ({model_result['confidence']:.2%})")
            
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
    except FileNotFoundError:
        print(f"❌ Image file not found: {test_image_path}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_webapp_status():
    """Check if webapp is running"""
    try:
        response = requests.get("http://localhost:5000")
        if response.status_code == 200:
            print("✅ Webapp is running on localhost:5000")
            return True
        else:
            print(f"❌ Webapp returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Webapp is not running. Start it with 'python app.py'")
        return False

if __name__ == "__main__":
    print("🧪 OCT Webapp Compare Models Debug Tool")
    print("=" * 50)
    
    if test_webapp_status():
        test_compare_models()
    else:
        print("\n💡 To start the webapp:")
        print("   cd C:\\Users\\parim\\projects\\oct-analysis\\webapp")
        print("   python app.py")
