"""
Preprocessing utilities for chest X-ray images.
Must exactly match training-time preprocessing to ensure model compatibility.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configuration constants (must match training)
IMG_SIZE = 300

def medical_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE preprocessing exactly as used during training.

    This function applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    in the LAB color space to enhance medical image quality while preserving
    color information. This preprocessing was used during model training and
    must be applied identically for inference.

    Args:
        img: Input image as numpy array (RGB format, uint8)

    Returns:
        Preprocessed image ready for model input (float32, normalized)
    """
    # Ensure input is uint8 for OpenCV operations
    img_uint8 = img.astype("uint8")

    # Convert to LAB color space for CLAHE application
    # LAB separates luminance (L) from color channels (A,B)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

    # Split into individual channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to luminance channel only
    # Parameters match training: clipLimit=2.0, tileGridSize=(8,8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge enhanced luminance with original color channels
    img_enhanced = cv2.merge((l_enhanced, a, b))

    # Convert back to RGB color space
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)

    # Apply Keras preprocessing (normalization for EfficientNet)
    # This step is crucial for model compatibility
    return preprocess_input(img_rgb.astype("float32"))

def preprocess_for_inference(img: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline for inference.

    Args:
        img: Input image as numpy array (RGB format)

    Returns:
        Preprocessed image with batch dimension added, ready for model input
    """
    # Apply medical preprocessing
    processed_img = medical_preprocess(img)

    # Add batch dimension
    return np.expand_dims(processed_img, axis=0)

def resize_image(img: np.ndarray, target_size: tuple = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """
    Resize image using PIL and convert to float32 (EXACTLY matches Keras img_to_array).
    
    CRITICAL: Keras img_to_array returns float32 in range [0, 255], not uint8!
    This function must replicate that exact behavior.

    Args:
        img: Input image as numpy array
        target_size: Target size as (width, height) tuple

    Returns:
        Resized image as numpy array in float32 format (matches Keras)
    """
    from PIL import Image
    from tensorflow.keras.preprocessing.image import img_to_array
    
    # Convert numpy array to PIL Image
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img
    
    pil_img = Image.fromarray(img_uint8)
    
    # Resize using PIL with LANCZOS (same as ImageDataGenerator target_size)
    resized_pil = pil_img.resize(target_size, Image.LANCZOS)
    
    # Convert to array using Keras function (returns float32, NOT uint8!)
    # This matches EXACTLY what ImageDataGenerator does
    return img_to_array(resized_pil)
