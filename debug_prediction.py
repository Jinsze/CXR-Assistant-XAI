"""
Debug script to test if predictions match between notebook and web app
"""
import cv2
import numpy as np
import tensorflow as tf
from lung_disease_ai.utils.preprocessing import resize_image, medical_preprocess, IMG_SIZE

# Load model
model = tf.keras.models.load_model('lung_disease_ai/models/lung_model.keras')

# Class mapping from training
class_indices = {'covid': 0, 'normal': 1, 'pneumonia': 2, 'tuberculosis': 3}
class_labels = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

print("="*60)
print("PREDICTION DEBUG SCRIPT")
print("="*60)

# Test image path - REPLACE THIS with path to your tuberculosis test image
# test_image_path = "path/to/your/tuberculosis_image.png"

# For now, create a dummy test
dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

print(f"\n1. Original image shape: {dummy_img.shape}")

# EXACT preprocessing as in training
print("\n2. Applying preprocessing (resize → CLAHE → normalize)...")

# Step 1: Resize to 300x300
resized = resize_image(dummy_img, (IMG_SIZE, IMG_SIZE))
print(f"   After resize: {resized.shape}")

# Step 2: Apply CLAHE preprocessing
preprocessed = medical_preprocess(resized)
print(f"   After CLAHE: {preprocessed.shape}, dtype: {preprocessed.dtype}")
print(f"   Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")

# Step 3: Add batch dimension
batch_img = np.expand_dims(preprocessed, axis=0)
print(f"   Final batch shape: {batch_img.shape}")

# Predict
print("\n3. Running prediction...")
predictions = model.predict(batch_img, verbose=0)

if isinstance(predictions, list):
    predictions = predictions[0]

pred_probs = predictions[0]
pred_idx = np.argmax(pred_probs)

print("\n4. Results:")
print(f"   Predicted index: {pred_idx}")
print(f"   Predicted class: {class_labels[pred_idx]}")
print(f"   Confidence: {pred_probs[pred_idx]:.2%}")

print("\n5. All probabilities:")
for idx, (label, prob) in enumerate(zip(class_labels, pred_probs)):
    print(f"   [{idx}] {label:15s}: {prob:.4f} ({prob*100:.2f}%)")

print("\n" + "="*60)
print("TO USE THIS SCRIPT WITH YOUR TEST IMAGE:")
print("1. Uncomment the test_image_path line")
print("2. Replace dummy_img with: cv2.imread(test_image_path)")
print("3. Convert to RGB: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
print("="*60)

