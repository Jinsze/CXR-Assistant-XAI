"""
QUICK TEST: Put your TB image here and run this to see the difference!
"""

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# CHANGE THIS to your TB image path!
IMAGE_PATH = "test_tb.png"  # ← Change this!

print("="*80)
print("QUICK DIAGNOSIS TEST")
print("="*80)

try:
    # Method 1: Web App (PIL)
    print("\n1. WEB APP METHOD:")
    img_pil = Image.open(IMAGE_PATH)
    print(f"   Original mode: {img_pil.mode}")
    
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        print(f"   Converted to: RGB")
    
    img_array_webapp = np.array(img_pil)
    print(f"   Shape: {img_array_webapp.shape}")
    print(f"   Dtype: {img_array_webapp.dtype}")
    print(f"   Range: [{img_array_webapp.min()}, {img_array_webapp.max()}]")
    print(f"   First pixel: {img_array_webapp[0,0]}")
    
    # Method 2: Training (Keras)
    print("\n2. TRAINING METHOD (Keras):")
    img_keras = load_img(IMAGE_PATH)
    img_array_keras = img_to_array(img_keras)
    print(f"   Shape: {img_array_keras.shape}")
    print(f"   Dtype: {img_array_keras.dtype}")
    print(f"   Range: [{img_array_keras.min()}, {img_array_keras.max()}]")
    print(f"   First pixel: {img_array_keras[0,0]}")
    
    # Comparison
    print("\n3. COMPARISON:")
    # Convert webapp to float32 like Keras
    img_webapp_float = img_array_webapp.astype('float32')
    diff = np.abs(img_webapp_float - img_array_keras).mean()
    print(f"   Mean difference: {diff}")
    
    if diff < 0.001:
        print("   ✅ PERFECT MATCH!")
    else:
        print("   ❌ DIFFERENT!")
        print(f"   Web app dtype: {img_array_webapp.dtype}")
        print(f"   Keras dtype: {img_array_keras.dtype}")

except FileNotFoundError:
    print(f"\n❌ Image not found: {IMAGE_PATH}")
    print("\nPLEASE:")
    print("1. Copy your TB test image to lung_disease_ai folder")
    print("2. Edit this file and change IMAGE_PATH to match your filename")
    print("3. Run: python QUICK_FIX_TEST.py")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

