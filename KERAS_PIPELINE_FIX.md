# Critical Fix: Using Keras img_to_array for Exact Pipeline Matching

## Date: December 23, 2025

## Problem
Despite fixing RGB format and resize interpolation, predictions were still incorrect (predicting COVID-19 instead of Tuberculosis for TB images).

## Root Cause
**The conversion from PIL Image to numpy array was not matching the training pipeline exactly.**

### Training Pipeline (ImageDataGenerator)
```python
# What ImageDataGenerator does internally:
1. Load image → PIL Image in RGB
2. Resize to target_size → PIL.resize(..., LANCZOS)
3. Convert to array → keras.preprocessing.image.img_to_array()
4. Apply preprocessing_function → medical_preprocess()
```

### Previous Web App (WRONG)
```python
# What we were doing:
1. Load image → PIL Image
2. Convert to RGB → image.convert('RGB')
3. Convert to numpy → np.array(image)  # ❌ NOT the same as img_to_array!
4. Resize → PIL resize
5. Apply preprocessing → medical_preprocess()
```

### The Critical Difference
- `np.array(image)` returns a `uint8` array with values [0, 255]
- `keras.preprocessing.image.img_to_array(image)` returns a `float32` array with values [0, 255]
- Even though both have the same values, the **dtype matters** because:
  - When medical_preprocess does `.astype("uint8")`, it may handle float32 → uint8 conversion differently
  - Potential rounding/truncation differences affect CLAHE calculations

## The Fix

### New Pipeline (CORRECT)
```python
def preprocess_image_for_model(pil_image) -> np.ndarray:
    """Use Keras functions directly to match training exactly."""
    from tensorflow.keras.preprocessing.image import img_to_array
    
    # Step 1: Resize using PIL (same as ImageDataGenerator)
    resized_pil = pil_image.resize((300, 300), Image.LANCZOS)
    
    # Step 2: Convert to array using Keras function (returns float32)
    img_array = img_to_array(resized_pil)  # ✅ Exact match!
    
    # Step 3: Apply medical preprocessing
    preprocessed = medical_preprocess(img_array)
    
    # Step 4: Add batch dimension
    return np.expand_dims(preprocessed, axis=0)
```

## Changes Made

### 1. `app.py`
- Modified `preprocess_image_for_model` to accept PIL image directly
- Use `img_to_array` instead of `np.array`
- Updated `analyze_image` to accept and pass PIL image
- Pass separate original image to Grad-CAM for display

### 2. `utils/gradcam.py`
- Modified `generate_explanation_visualization` to accept both:
  - `preprocessed_image`: for model inference
  - `original_image`: for display overlay
- This prevents display artifacts from normalized images

## Verification Steps

1. **Test with TB image**:
   - Should now predict "tuberculosis" with high confidence
   - Should match notebook predictions exactly

2. **Test with other classes**:
   - COVID-19, Normal, Pneumonia images
   - Verify confidence scores match training/notebook

3. **Check Grad-CAM**:
   - Should display clear, colorful heatmaps
   - Should highlight relevant lung regions

## Technical Details

### Why img_to_array Matters
```python
# np.array behavior
img = Image.open('xray.jpg')  # PIL Image, mode RGB
arr1 = np.array(img)         # dtype: uint8, shape: (H, W, 3)

# img_to_array behavior  
arr2 = img_to_array(img)     # dtype: float32, shape: (H, W, 3)

# Both have same values, but different dtypes!
# When medical_preprocess does arr.astype('uint8'):
# - uint8 → uint8: no conversion
# - float32 → uint8: may round differently
```

### Complete Aligned Pipeline
```
┌─────────────────┐
│ Upload Image    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PIL.Image.open  │ (any mode: L, RGB, RGBA, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ .convert('RGB') │ (force RGB mode)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ .resize(300,300)│ (PIL LANCZOS interpolation)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ img_to_array()  │ (Keras function → float32)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ medical_preproc │ (CLAHE + normalize)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Predict   │
└─────────────────┘

This EXACTLY matches ImageDataGenerator!
```

## Expected Outcome
✅ Web app predictions should now match notebook predictions exactly  
✅ TB images should correctly predict "tuberculosis"  
✅ All confidence scores should match training environment  
✅ Grad-CAM should display properly on original images

