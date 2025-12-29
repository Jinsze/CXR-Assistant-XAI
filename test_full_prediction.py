"""
Complete prediction test - compares predictions between different loading methods.
This will show you exactly what's different.
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Import your utilities
from utils.preprocessing import medical_preprocess, resize_image, IMG_SIZE
from utils.predict import LungDiseaseClassifier

def medical_preprocess_copy(img):
    """Exact copy from training"""
    img_uint8 = img.astype("uint8")
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    img_enhanced = cv2.merge((l_enhanced, a, b))
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    return preprocess_input(img_rgb.astype("float32"))

def test_prediction_methods(image_path):
    """
    Test predictions using different image loading methods.
    """
    print("="*100)
    print(f"TESTING PREDICTIONS FOR: {image_path}")
    print("="*100)
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model('models/lung_model.keras')
    print("✅ Model loaded")
    
    class_labels = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
    
    # ========== METHOD 1: WEB APP (PIL with RGB) ==========
    print("\n" + "="*100)
    print("METHOD 1: WEB APP (PIL with .convert('RGB'))")
    print("="*100)
    
    img_pil = Image.open(image_path)
    print(f"Original mode: {img_pil.mode}")
    
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    
    img_array = np.array(img_pil)
    print(f"Shape after PIL load: {img_array.shape}")
    print(f"First pixel: {img_array[0,0]}")
    
    # Resize
    img_resized = resize_image(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Preprocess
    img_preprocessed = medical_preprocess(img_resized)
    print(f"After preprocessing - Mean: {img_preprocessed.mean():.4f}, Std: {img_preprocessed.std():.4f}")
    
    # Add batch dimension
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)
    if isinstance(predictions, list):
        predictions = predictions[0]
    pred_idx = np.argmax(predictions[0])
    
    print(f"\n📊 PREDICTIONS (Method 1 - Web App):")
    for i, (label, conf) in enumerate(zip(class_labels, predictions[0])):
        marker = "👉" if i == pred_idx else "  "
        print(f"{marker} {label:15s}: {conf:6.2%}")
    
    # ========== METHOD 2: KERAS load_img (Training equivalent) ==========
    print("\n" + "="*100)
    print("METHOD 2: KERAS load_img (ImageDataGenerator equivalent)")
    print("="*100)
    
    img_keras = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array_keras = img_to_array(img_keras)
    print(f"Shape after Keras load: {img_array_keras.shape}")
    print(f"First pixel: {img_array_keras[0,0]}")
    
    # Preprocess
    img_preprocessed_keras = medical_preprocess_copy(img_array_keras)
    print(f"After preprocessing - Mean: {img_preprocessed_keras.mean():.4f}, Std: {img_preprocessed_keras.std():.4f}")
    
    # Add batch dimension
    img_batch_keras = np.expand_dims(img_preprocessed_keras, axis=0)
    
    # Predict
    predictions_keras = model.predict(img_batch_keras, verbose=0)
    if isinstance(predictions_keras, list):
        predictions_keras = predictions_keras[0]
    pred_idx_keras = np.argmax(predictions_keras[0])
    
    print(f"\n📊 PREDICTIONS (Method 2 - Training Method):")
    for i, (label, conf) in enumerate(zip(class_labels, predictions_keras[0])):
        marker = "👉" if i == pred_idx_keras else "  "
        print(f"{marker} {label:15s}: {conf:6.2%}")
    
    # ========== METHOD 3: OpenCV imread (WRONG - BGR) ==========
    print("\n" + "="*100)
    print("METHOD 3: OpenCV cv2.imread (WRONG WAY - BGR)")
    print("="*100)
    
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is not None:
        print(f"Shape after cv2 load: {img_cv2.shape}")
        print(f"First pixel BGR: {img_cv2[0,0]}")
        print("⚠️  WARNING: This loads in BGR format!")
        
        # Resize
        img_resized_cv2 = cv2.resize(img_cv2, (IMG_SIZE, IMG_SIZE))
        
        # Preprocess (will be wrong because input is BGR not RGB)
        img_preprocessed_cv2 = medical_preprocess(img_resized_cv2)
        print(f"After preprocessing - Mean: {img_preprocessed_cv2.mean():.4f}, Std: {img_preprocessed_cv2.std():.4f}")
        
        # Add batch dimension
        img_batch_cv2 = np.expand_dims(img_preprocessed_cv2, axis=0)
        
        # Predict
        predictions_cv2 = model.predict(img_batch_cv2, verbose=0)
        if isinstance(predictions_cv2, list):
            predictions_cv2 = predictions_cv2[0]
        pred_idx_cv2 = np.argmax(predictions_cv2[0])
        
        print(f"\n📊 PREDICTIONS (Method 3 - WRONG WAY):")
        for i, (label, conf) in enumerate(zip(class_labels, predictions_cv2[0])):
            marker = "👉" if i == pred_idx_cv2 else "  "
            print(f"{marker} {label:15s}: {conf:6.2%}")
    
    # ========== COMPARISON ==========
    print("\n" + "="*100)
    print("COMPARISON & ANALYSIS")
    print("="*100)
    
    # Compare preprocessing arrays
    diff = np.abs(img_preprocessed - img_preprocessed_keras).mean()
    print(f"\n1. Preprocessing Difference (Web App vs Training):")
    print(f"   Mean absolute difference: {diff:.8f}")
    
    if diff < 1e-6:
        print("   ✅ PERFECT MATCH - Preprocessing is identical!")
    elif diff < 0.001:
        print("   ✅ VERY CLOSE - Minor floating point differences only")
    else:
        print("   ❌ SIGNIFICANT DIFFERENCE - This will cause prediction errors!")
    
    # Compare predictions
    pred_diff = np.abs(predictions[0] - predictions_keras[0]).mean()
    print(f"\n2. Prediction Difference (Web App vs Training):")
    print(f"   Mean absolute difference: {pred_diff:.8f}")
    
    if pred_diff < 0.001:
        print("   ✅ PREDICTIONS MATCH!")
    else:
        print("   ❌ PREDICTIONS DIFFER!")
        print(f"   Web App predicts: {class_labels[pred_idx]}")
        print(f"   Training method predicts: {class_labels[pred_idx_keras]}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    if pred_idx == pred_idx_keras and pred_diff < 0.001:
        print("✅ SUCCESS! Web app matches training method perfectly!")
        print(f"   Both predict: {class_labels[pred_idx]} ({predictions[0][pred_idx]:.2%})")
    else:
        print("❌ PROBLEM DETECTED!")
        print(f"   Web App: {class_labels[pred_idx]} ({predictions[0][pred_idx]:.2%})")
        print(f"   Training: {class_labels[pred_idx_keras]} ({predictions_keras[0][pred_idx_keras]:.2%})")
        print("\n   Possible causes:")
        print("   1. Image not being loaded in RGB format correctly")
        print("   2. Preprocessing order is different")
        print("   3. Model file mismatch")
        print("   4. Browser cache issue (try Ctrl+Shift+R)")

if __name__ == "__main__":
    import sys
    import os
    
    print("FULL PREDICTION TEST")
    print("="*100)
    print("This script tests if predictions match between web app and training methods.")
    print()
    
    # Check for test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for common test files
        test_files = ['test_tb.png', 'test.png', 'test.jpg', '../test_tb.png']
        image_path = None
        for f in test_files:
            if os.path.exists(f):
                image_path = f
                break
        
        if image_path is None:
            print("❌ No test image found!")
            print("\nUSAGE:")
            print("   python test_full_prediction.py <path_to_image>")
            print("\nExample:")
            print("   python test_full_prediction.py test_tb.png")
            sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Testing with image: {image_path}")
    print()
    
    test_prediction_methods(image_path)

