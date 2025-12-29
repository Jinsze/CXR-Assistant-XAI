"""
Test script to verify preprocessing matches training exactly.
Compare web app preprocessing vs ImageDataGenerator preprocessing.
"""

import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Import your preprocessing
from utils.preprocessing import medical_preprocess, resize_image, IMG_SIZE

def medical_preprocess_training(img):
    """
    Exact copy of your training preprocessing function.
    """
    img_uint8 = img.astype("uint8")
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    img_enhanced = cv2.merge((l_enhanced, a, b))
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    return preprocess_input(img_rgb.astype("float32"))

def test_image_path(image_path):
    """
    Test if web app preprocessing matches training preprocessing.
    """
    print("="*80)
    print(f"Testing image: {image_path}")
    print("="*80)
    
    # METHOD 1: Web App (current implementation)
    print("\n1. WEB APP METHOD (PIL with RGB conversion):")
    img_pil = Image.open(image_path)
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_array_webapp = np.array(img_pil)
    print(f"   - Loaded shape: {img_array_webapp.shape}")
    print(f"   - Data type: {img_array_webapp.dtype}")
    print(f"   - First pixel RGB: {img_array_webapp[0,0]}")
    
    # Resize
    img_resized_webapp = resize_image(img_array_webapp, (IMG_SIZE, IMG_SIZE))
    print(f"   - After resize: {img_resized_webapp.shape}")
    
    # Preprocess
    img_preprocessed_webapp = medical_preprocess(img_resized_webapp)
    print(f"   - After preprocessing: {img_preprocessed_webapp.shape}, dtype: {img_preprocessed_webapp.dtype}")
    print(f"   - Value range: [{img_preprocessed_webapp.min():.3f}, {img_preprocessed_webapp.max():.3f}]")
    print(f"   - Mean: {img_preprocessed_webapp.mean():.3f}, Std: {img_preprocessed_webapp.std():.3f}")
    
    # METHOD 2: Training Method (Keras load_img)
    print("\n2. TRAINING METHOD (Keras load_img - ImageDataGenerator equivalent):")
    img_keras = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array_keras = img_to_array(img_keras)
    print(f"   - Loaded shape: {img_array_keras.shape}")
    print(f"   - Data type: {img_array_keras.dtype}")
    print(f"   - First pixel RGB: {img_array_keras[0,0]}")
    
    # Preprocess
    img_preprocessed_keras = medical_preprocess_training(img_array_keras)
    print(f"   - After preprocessing: {img_preprocessed_keras.shape}, dtype: {img_preprocessed_keras.dtype}")
    print(f"   - Value range: [{img_preprocessed_keras.min():.3f}, {img_preprocessed_keras.max():.3f}]")
    print(f"   - Mean: {img_preprocessed_keras.mean():.3f}, Std: {img_preprocessed_keras.std():.3f}")
    
    # METHOD 3: WRONG WAY (cv2.imread - BGR)
    print("\n3. WRONG METHOD (cv2.imread - BGR format):")
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is not None:
        print(f"   - Loaded shape: {img_cv2.shape}")
        print(f"   - Data type: {img_cv2.dtype}")
        print(f"   - First pixel BGR: {img_cv2[0,0]}")
        print(f"   ⚠️  NOTE: Color channels are REVERSED (BGR not RGB)!")
    
    # COMPARISON
    print("\n" + "="*80)
    print("COMPARISON:")
    print("="*80)
    diff = np.abs(img_preprocessed_webapp - img_preprocessed_keras).mean()
    print(f"Mean absolute difference (Web App vs Training): {diff:.6f}")
    
    if diff < 0.001:
        print("✅ PERFECT MATCH! Preprocessing is identical.")
    elif diff < 0.01:
        print("✅ VERY CLOSE! Minor floating-point differences only.")
    else:
        print("❌ MISMATCH! Preprocessing differs significantly!")
        print("   This will cause prediction errors!")
    
    # Check if arrays are exactly equal
    if np.allclose(img_preprocessed_webapp, img_preprocessed_keras, atol=1e-5):
        print("✅ Arrays are numerically equivalent (within tolerance).")
    else:
        print("❌ Arrays differ beyond numerical tolerance!")
    
    return img_preprocessed_webapp, img_preprocessed_keras

if __name__ == "__main__":
    print("PREPROCESSING VERIFICATION TOOL")
    print("="*80)
    print()
    print("This tool compares your web app preprocessing with training preprocessing.")
    print()
    print("USAGE:")
    print("1. Place a test image in this directory (e.g., 'test_tb.png')")
    print("2. Run: python test_preprocessing.py")
    print()
    
    # Test with your tuberculosis image
    import os
    test_files = ['test_tb.png', 'test.png', 'sample.png', 'test.jpg']
    
    found = False
    for test_file in test_files:
        if os.path.exists(test_file):
            found = True
            webapp, training = test_image_path(test_file)
            break
    
    if not found:
        print("⚠️  No test image found!")
        print(f"   Place one of these files in the directory: {test_files}")
        print()
        print("To test your TB image:")
        print("1. Copy your TB test image to this directory")
        print("2. Rename it to 'test_tb.png'")
        print("3. Run this script again")

