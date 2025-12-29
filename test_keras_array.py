"""
Test script to verify img_to_array produces the same results as ImageDataGenerator.
"""
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from utils.preprocessing import medical_preprocess

# Path to test image (update this to your actual TB image path)
# Example paths - update with your actual path:
# TEST_IMAGE_PATH = r"C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\test_images\TB.1040.jpg"
# TEST_IMAGE_PATH = r"C:\path\to\your\TB.1040.jpg"
TEST_IMAGE_PATH = r"C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\lung_disease_ai\assets\sample_xray.jpg"  # Placeholder

print("=" * 80)
print("TESTING KERAS ARRAY CONVERSION")
print("=" * 80)

# Method 1: Using load_img (what ImageDataGenerator uses internally)
print("\n1. Using tf.keras.utils.load_img (ImageDataGenerator method):")
img1 = load_img(TEST_IMAGE_PATH, target_size=(300, 300))
arr1 = img_to_array(img1)
print(f"   Shape: {arr1.shape}, Dtype: {arr1.dtype}")
print(f"   Value range: [{arr1.min():.2f}, {arr1.max():.2f}]")

# Method 2: Using PIL + img_to_array (our new web app method)
print("\n2. Using PIL + img_to_array (New Web App method):")
img2 = Image.open(TEST_IMAGE_PATH)
if img2.mode != 'RGB':
    img2 = img2.convert('RGB')
img2_resized = img2.resize((300, 300), Image.LANCZOS)
arr2 = img_to_array(img2_resized)
print(f"   Shape: {arr2.shape}, Dtype: {arr2.dtype}")
print(f"   Value range: [{arr2.min():.2f}, {arr2.max():.2f}]")

# Method 3: Using np.array (OLD WRONG METHOD)
print("\n3. Using PIL + np.array (OLD WRONG method):")
img3 = Image.open(TEST_IMAGE_PATH)
if img3.mode != 'RGB':
    img3 = img3.convert('RGB')
img3_resized = img3.resize((300, 300), Image.LANCZOS)
arr3 = np.array(img3_resized)
print(f"   Shape: {arr3.shape}, Dtype: {arr3.dtype}")
print(f"   Value range: [{arr3.min():.2f}, {arr3.max():.2f}]")

# Compare arrays
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"\nMethod 1 vs Method 2 (should be IDENTICAL):")
print(f"  Arrays equal: {np.allclose(arr1, arr2)}")
print(f"  Max difference: {np.abs(arr1 - arr2).max()}")

print(f"\nMethod 1 vs Method 3 (DIFFERENT dtypes):")
print(f"  Values equal: {np.allclose(arr1, arr3)}")
print(f"  Dtype match: {arr1.dtype == arr3.dtype}")

# Test preprocessing
print("\n" + "=" * 80)
print("TESTING PREPROCESSING")
print("=" * 80)

proc1 = medical_preprocess(arr1)
proc2 = medical_preprocess(arr2)
proc3 = medical_preprocess(arr3)

print(f"\nPreprocessed Method 1 (load_img + img_to_array):")
print(f"  Shape: {proc1.shape}, Dtype: {proc1.dtype}")
print(f"  Value range: [{proc1.min():.4f}, {proc1.max():.4f}]")

print(f"\nPreprocessed Method 2 (PIL + img_to_array):")
print(f"  Shape: {proc2.shape}, Dtype: {proc2.dtype}")
print(f"  Value range: [{proc2.min():.4f}, {proc2.max():.4f}]")

print(f"\nPreprocessed Method 3 (PIL + np.array - WRONG):")
print(f"  Shape: {proc3.shape}, Dtype: {proc3.dtype}")
print(f"  Value range: [{proc3.min():.4f}, {proc3.max():.4f}]")

print(f"\nPreprocessed 1 vs 2 (should be IDENTICAL):")
print(f"  Arrays equal: {np.allclose(proc1, proc2)}")
print(f"  Max difference: {np.abs(proc1 - proc2).max()}")

print(f"\nPreprocessed 1 vs 3 (DIFFERENT):")
print(f"  Arrays equal: {np.allclose(proc1, proc3)}")
print(f"  Max difference: {np.abs(proc1 - proc3).max()}")

# Test with model
print("\n" + "=" * 80)
print("TESTING WITH MODEL")
print("=" * 80)

from utils.predict import LungDiseaseClassifier

classifier = LungDiseaseClassifier()
if classifier.load_model():
    # Test all three methods
    pred1 = classifier.model.predict(np.expand_dims(proc1, axis=0), verbose=0)[0]
    pred2 = classifier.model.predict(np.expand_dims(proc2, axis=0), verbose=0)[0]
    pred3 = classifier.model.predict(np.expand_dims(proc3, axis=0), verbose=0)[0]
    
    class_labels = ['covid', 'normal', 'pneumonia', 'tuberculosis']
    
    print("\nPredictions with Method 1 (ImageDataGenerator method):")
    for label, conf in zip(class_labels, pred1):
        print(f"  {label}: {conf*100:.2f}%")
    print(f"  Predicted: {class_labels[np.argmax(pred1)]}")
    
    print("\nPredictions with Method 2 (New Web App - should MATCH above!):")
    for label, conf in zip(class_labels, pred2):
        print(f"  {label}: {conf*100:.2f}%")
    print(f"  Predicted: {class_labels[np.argmax(pred2)]}")
    
    print("\nPredictions with Method 3 (OLD WRONG method):")
    for label, conf in zip(class_labels, pred3):
        print(f"  {label}: {conf*100:.2f}%")
    print(f"  Predicted: {class_labels[np.argmax(pred3)]}")
    
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if np.allclose(pred1, pred2):
        print("✅ SUCCESS! Method 2 (New Web App) matches ImageDataGenerator!")
    else:
        print("❌ FAILED! Method 2 predictions differ from ImageDataGenerator")
        
    if not np.allclose(pred1, pred3):
        print("✅ CONFIRMED! Method 3 (OLD method) produces different predictions")
    else:
        print("⚠️  WARNING: Method 3 predictions match (unexpected)")
else:
    print("❌ Failed to load model!")

print("\n" + "=" * 80)

