from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Create a simple test image
from PIL import Image
test_img = Image.new('RGB', (100, 100), color='red')
test_img.save('test_temp.png')

# Check what Keras returns
img = load_img('test_temp.png', target_size=(300, 300))
arr = img_to_array(img)
print(f"Keras img_to_array dtype: {arr.dtype}")
print(f"Keras img_to_array range: [{arr.min():.1f}, {arr.max():.1f}]")
print(f"Keras img_to_array shape: {arr.shape}")

# Check what PIL returns
import numpy as np
arr_pil = np.array(img)
print(f"\nPIL np.array dtype: {arr_pil.dtype}")
print(f"PIL np.array range: [{arr_pil.min()}, {arr_pil.max()}]")
print(f"PIL np.array shape: {arr_pil.shape}")

import os
os.remove('test_temp.png')

