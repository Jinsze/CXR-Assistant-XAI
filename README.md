---
title: CXR-Assistant
emoji: 🫁
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# 🫁 CXR-Assistant: Explainable AI for Chest X-ray Analysis

**Deep Learning System for Multi-Disease Detection**

An advanced AI-powered web application for automated chest X-ray analysis, featuring deep learning classification and explainable AI visualization. Developed as part of a Data Science project at Universiti Malaya by Liew Jin Sze.

## 🎯 Project Overview

CXR-Assistant is a research-grade medical imaging tool that combines state-of-the-art deep learning with clinical interpretability. The system analyzes chest X-ray images to detect multiple lung conditions while providing visual explanations of the AI's decision-making process through Grad-CAM++ heatmaps.

### Key Features

- **🔬 Multi-Disease Classification**: Detects COVID-19, Pneumonia, Tuberculosis, and Normal cases
- **🧠 EfficientNet-B3 Architecture**: State-of-the-art transfer learning model
- **✨ CLAHE Enhancement**: Clinical-grade image preprocessing for optimal feature extraction
- **📊 Grad-CAM++ Explainability**: Visual explanations showing which lung regions influenced the prediction
- **🎯 Temperature Scaling**: Post-hoc calibration (T=1.1013) for reliable confidence scores
- **💻 Interactive Web Interface**: User-friendly Streamlit application with professional medical UI

## 🚀 Try It Out

1. **Upload** a chest X-ray image (JPG/PNG format)
2. **Click "Analyze X-Ray"** to process the image
3. **View Results**: Get AI diagnosis with calibrated confidence scores
4. **Explore Grad-CAM++**: See which lung regions the AI focused on for its decision

## 📊 Model Performance

The model was trained and validated on a diverse chest X-ray dataset:

- **Overall Accuracy**: 98%
- **COVID-19**: Precision 0.96, Recall 0.97, F1 0.97
- **Normal**: Precision 0.99, Recall 0.98, F1 0.98
- **Pneumonia**: Precision 0.94, Recall 0.98, F1 0.96
- **Tuberculosis**: Precision 1.00, Recall 1.00, F1 1.00
- **ROC-AUC Score**: 0.9991

## 🔬 Technical Details

### Architecture
- **Base Model**: EfficientNet-B3 (pre-trained on ImageNet)
- **Input Size**: 300×300 pixels
- **Output**: 4-class softmax with temperature scaling
- **Framework**: TensorFlow 2.13+ / Keras

### Preprocessing Pipeline
1. **Image Loading**: Keras image utilities (PIL backend)
2. **Resizing**: 300×300 pixels using bilinear interpolation
3. **CLAHE Enhancement**: 
   - LAB color space conversion
   - Contrast Limited Adaptive Histogram Equalization (clipLimit=2.0, tileGridSize=8×8)
   - Applied to L-channel only
4. **Normalization**: EfficientNet-specific preprocessing

### Calibration
- **Method**: Post-hoc Temperature Scaling
- **Temperature**: T = 1.1013 (optimized on validation set)
- **Benefit**: Improved Expected Calibration Error (ECE) for clinical reliability

### Explainable AI
- **Method**: Grad-CAM++ (Gradient-weighted Class Activation Mapping++)
- **Target Layer**: `top_conv` (final convolutional layer)
- **Visualization**: Jet colormap overlay on original image
- **Purpose**: Clinical transparency and trust in AI predictions

## 📁 Project Structure

```
lung_disease_ai/
├── app.py                      # Main Streamlit application
├── models/
│   └── colab_clahe_eff_final.keras  # Trained EfficientNet-B3 model
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # CLAHE + medical preprocessing
│   ├── predict.py             # Model loading and inference
│   └── gradcam.py             # Grad-CAM++ implementation
├── assets/
│   └── sample_xray.png        # Sample chest X-ray images
├── requirements.txt           # Python dependencies
├── .gitattributes            # Git LFS configuration
└── README.md                 # This file
```

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- TensorFlow 2.13+
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/CXR-Assistant
cd CXR-Assistant

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Model File
The trained model (`colab_clahe_eff_final.keras`) is tracked using Git LFS due to its size. Make sure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

## ⚠️ Important Disclaimers

### Medical Use
**This tool is for academic research and educational purposes ONLY.**

- ❌ NOT approved for clinical diagnosis or patient care
- ❌ NOT a replacement for professional medical expertise
- ❌ NOT validated on all populations or imaging protocols
- ✅ For research, education, and AI explainability demonstrations

### Limitations
- Performance may vary with image quality, patient demographics, and X-ray protocols
- The model was trained on specific datasets and may not generalize to all clinical settings
- Always consult qualified healthcare professionals for medical decisions
- AI systems can make errors; human oversight is essential

## 🎓 Academic Context

**Project**: Data Science Final Project  
**Institution**: Universiti Malaya  
**Developer**: Liew Jin Sze  
**Focus Areas**: Medical Imaging, Deep Learning, Explainable AI, Clinical Decision Support

This project demonstrates the application of modern AI techniques to medical imaging challenges, with emphasis on interpretability and clinical reliability through temperature scaling and Grad-CAM++ visualizations.

## 📖 References

### Key Papers
1. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019)
2. **Grad-CAM++**: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks" (WACV 2018)
3. **Temperature Scaling**: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
4. **CLAHE**: Zuiderveld, "Contrast Limited Adaptive Histogram Equalization" (Graphics Gems 1994)

## 🔧 Configuration

### Updating Model Path
Edit `utils/predict.py`:
```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'colab_clahe_eff_final.keras')
```

### Adjusting Temperature
Edit `app.py`:
```python
CALIBRATION_TEMPERATURE = 1.1013  # Modify based on your validation results
```

## 🤝 Contributing

This is an academic project. For suggestions or collaborations:
1. Open an issue on the repository
2. Describe your proposed enhancement
3. Academic collaborations are welcome

## 📄 License

MIT License - Free for academic and research use. Please cite appropriately if used in publications.

## 📞 Contact

For academic inquiries or technical questions:
- **Developer**: Liew Jin Sze
- **Institution**: Universiti Malaya
- **Project Type**: Data Science Final Project

---

**Version**: CXR-Assistant v1.0  
**Last Updated**: December 2025  
**Status**: 🟢 Active Research Project

**Powered by**: EfficientNet-B3 • CLAHE Enhancement • Grad-CAM++ Explainability • Temperature Scaling

*Remember: AI should augment, not replace, medical expertise. Always prioritize patient safety and ethical practice.*
