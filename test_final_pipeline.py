"""
Final test script to verify the web app preprocessing matches test flow exactly.
This script compares the web app method with the notebook test flow method.
"""
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Import web app functions
from utils.preprocessing import medical_preprocess
from utils.predict import LungDiseaseClassifier

# Configuration
IMG_SIZE = 300

# === UPDATE THIS PATH TO YOUR TEST IMAGE ===
TEST_IMAGE_PATH = r"C:\path\to\your\TB.1040.jpg"  # UPDATE THIS!

print("=" * 100)
print("FINAL PIPELINE VERIFICATION")
print("=" * 100)
print(f"\nTest Image: {TEST_IMAGE_PATH}")

# ============================================================================
# METHOD 1: NOTEBOOK TEST FLOW (from clahe_eff.ipynb)
# ============================================================================
print("\n" + "=" * 100)
print("METHOD 1: NOTEBOOK TEST FLOW (clahe_eff.ipynb)")
print("=" * 100)

print("\n📋 Steps:")
print("   1. cv2.imread → BGR")
print("   2. cv2.cvtColor(BGR → RGB)")
print("   3. cv2.resize((300, 300)) with LINEAR interpolation")
print("   4. medical_preprocess()")
print("   5. Add batch dimension")

# Step 1: Load with cv2 (BGR)
raw_img = cv2.imread(TEST_IMAGE_PATH)
print(f"\n✓ After cv2.imread: shape={raw_img.shape}, dtype={raw_img.dtype}")

# Step 2: Convert BGR to RGB
rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
print(f"✓ After BGR→RGB: shape={rgb_img.shape}, dtype={rgb_img.dtype}")
print(f"  First pixel: {rgb_img[0, 0]}")

# Step 3: Resize using cv2 (LINEAR interpolation by default)
resized_img = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
print(f"✓ After cv2.resize: shape={resized_img.shape}, dtype={resized_img.dtype}")
print(f"  First pixel after resize: {resized_img[0, 0]}")

# Step 4: Apply medical preprocessing
preprocessed_1 = medical_preprocess(resized_img)
print(f"✓ After medical_preprocess: shape={preprocessed_1.shape}, dtype={preprocessed_1.dtype}")
print(f"  Value range: [{preprocessed_1.min():.4f}, {preprocessed_1.max():.4f}]")
print(f"  Mean: {preprocessed_1.mean():.4f}, Std: {preprocessed_1.std():.4f}")

# Step 5: Add batch dimension
input_tensor_1 = np.expand_dims(preprocessed_1, axis=0)
print(f"✓ Final tensor: shape={input_tensor_1.shape}")

# ============================================================================
# METHOD 2: WEB APP FLOW (Current Implementation)
# ============================================================================
print("\n" + "=" * 100)
print("METHOD 2: WEB APP FLOW (app.py)")
print("=" * 100)

print("\n📋 Steps:")
print("   1. PIL.Image.open → RGB mode")
print("   2. .convert('RGB') if needed")
print("   3. np.array() → numpy RGB")
print("   4. cv2.resize((300, 300)) with LINEAR interpolation")
print("   5. medical_preprocess()")
print("   6. Add batch dimension")

# Step 1 & 2: Load with PIL and ensure RGB
pil_img = Image.open(TEST_IMAGE_PATH)
print(f"\n✓ After PIL.open: mode={pil_img.mode}, size={pil_img.size}")

if pil_img.mode != 'RGB':
    pil_img = pil_img.convert('RGB')
    print(f"✓ Converted to RGB mode")

# Step 3: Convert to numpy array
rgb_img_webapp = np.array(pil_img)
print(f"✓ After np.array: shape={rgb_img_webapp.shape}, dtype={rgb_img_webapp.dtype}")
print(f"  First pixel: {rgb_img_webapp[0, 0]}")

# Step 4: Resize using cv2 (LINEAR interpolation)
resized_img_webapp = cv2.resize(rgb_img_webapp, (IMG_SIZE, IMG_SIZE))
print(f"✓ After cv2.resize: shape={resized_img_webapp.shape}, dtype={resized_img_webapp.dtype}")
print(f"  First pixel after resize: {resized_img_webapp[0, 0]}")

# Step 5: Apply medical preprocessing
preprocessed_2 = medical_preprocess(resized_img_webapp)
print(f"✓ After medical_preprocess: shape={preprocessed_2.shape}, dtype={preprocessed_2.dtype}")
print(f"  Value range: [{preprocessed_2.min():.4f}, {preprocessed_2.max():.4f}]")
print(f"  Mean: {preprocessed_2.mean():.4f}, Std: {preprocessed_2.std():.4f}")

# Step 6: Add batch dimension
input_tensor_2 = np.expand_dims(preprocessed_2, axis=0)
print(f"✓ Final tensor: shape={input_tensor_2.shape}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 100)
print("PREPROCESSING COMPARISON")
print("=" * 100)

print(f"\nResized images equal: {np.array_equal(resized_img, resized_img_webapp)}")
print(f"Resized images close: {np.allclose(resized_img, resized_img_webapp)}")
if not np.array_equal(resized_img, resized_img_webapp):
    diff = np.abs(resized_img.astype(float) - resized_img_webapp.astype(float))
    print(f"  Max difference: {diff.max()}")
    print(f"  Mean difference: {diff.mean():.6f}")

print(f"\nPreprocessed arrays equal: {np.array_equal(preprocessed_1, preprocessed_2)}")
print(f"Preprocessed arrays close: {np.allclose(preprocessed_1, preprocessed_2)}")
if not np.array_equal(preprocessed_1, preprocessed_2):
    diff = np.abs(preprocessed_1 - preprocessed_2)
    print(f"  Max difference: {diff.max()}")
    print(f"  Mean difference: {diff.mean():.6f}")

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================
print("\n" + "=" * 100)
print("MODEL PREDICTIONS")
print("=" * 100)

classifier = LungDiseaseClassifier()
if classifier.load_model():
    print("✓ Model loaded successfully")
    
    # Predict with both methods
    preds_1 = classifier.model.predict(input_tensor_1, verbose=0)
    preds_2 = classifier.model.predict(input_tensor_2, verbose=0)
    
    # Handle list output
    if isinstance(preds_1, list):
        preds_1 = preds_1[0]
    if isinstance(preds_2, list):
        preds_2 = preds_2[0]
    
    preds_1 = preds_1[0]
    preds_2 = preds_2[0]
    
    class_labels = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
    
    print("\n📊 METHOD 1 (Notebook Test Flow):")
    for label, conf in zip(class_labels, preds_1):
        print(f"   {label:15s}: {conf*100:6.2f}%")
    print(f"   → Predicted: {class_labels[np.argmax(preds_1)]}")
    
    print("\n📊 METHOD 2 (Web App):")
    for label, conf in zip(class_labels, preds_2):
        print(f"   {label:15s}: {conf*100:6.2f}%")
    print(f"   → Predicted: {class_labels[np.argmax(preds_2)]}")
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    
    predictions_match = np.allclose(preds_1, preds_2, atol=1e-6)
    
    if predictions_match:
        print("\n✅ SUCCESS! Web app predictions MATCH notebook test flow!")
        print("✅ Preprocessing pipeline is IDENTICAL!")
        print("✅ The web app will produce the same results as your notebook!")
    else:
        print("\n❌ MISMATCH! Predictions differ:")
        diff = np.abs(preds_1 - preds_2)
        print(f"   Max prediction difference: {diff.max():.6f}")
        for i, (label, diff_val) in enumerate(zip(class_labels, diff)):
            if diff_val > 1e-6:
                print(f"   {label}: {diff_val:.6f} difference")
        print("\n🔍 Debug needed - check intermediate preprocessing steps above")
else:
    print("❌ Failed to load model!")

print("\n" + "=" * 100)

