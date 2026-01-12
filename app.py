"""
AI-Powered Lung Disease Classification System
Professional Streamlit web application with tabbed interface for chest X-ray analysis.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from typing import Optional, Tuple
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from io import BytesIO

# Import utility modules
from utils.preprocessing import resize_image, medical_preprocess, IMG_SIZE
from utils.predict import LungDiseaseClassifier
from utils.gradcam import generate_explanation_visualization

# Page configuration
st.set_page_config(
    page_title="Chest X-ray Assist: Explainable AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced professional medical styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced header with gradient */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Enhanced disclaimer with gradient background */
    .disclaimer {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        color: #92400e;
        box-shadow: 0 4px 6px rgba(245, 158, 11, 0.1);
        font-weight: 500;
    }
    
    /* Animated prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 15px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    
    .metric-card h3 {
        color: #1e40af;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* Modern tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 12px 12px 0 0;
        padding: 0 28px;
        font-weight: 600;
        font-size: 1rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: #3b82f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border-color: #1e40af;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4);
        transform: translateY(-2px);
    }
    
    /* Info boxes with rounded corners */
    .stInfo, .stSuccess, .stWarning {
        border-radius: 12px;
        border-left-width: 5px;
    }
    
    /* DataFrames with shadow */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* File uploader enhancement */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    }
    
    /* Image display with rounded corners */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Enhanced footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        color: #475569;
        font-size: 0.95rem;
        box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Temperature Scaling Configuration
CALIBRATION_TEMPERATURE = 1.1013  # Optimal temperature calculated via research methodology

def apply_temperature_scaling(raw_predictions: np.ndarray, temperature: float = CALIBRATION_TEMPERATURE) -> np.ndarray:
    """
    Apply Temperature Scaling with numerically stable logit-scaling.
    
    This implementation follows the exact steps for clinical calibration:
    1. Extract raw probability outputs from model
    2. Convert to log-space (logits)
    3. Scale by temperature T = 1.1013
    4. Apply numerical stability fix (subtract max before exp)
    5. Re-apply softmax for calibrated probabilities
    
    Args:
        raw_predictions: Raw probability outputs from model (shape: [num_classes])
        temperature: Calibration temperature T (default: 1.1013)
    
    Returns:
        Calibrated probability distribution (shape: [num_classes])
    
    References:
        - Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
        - Temperature > 1 reduces overconfidence by "softening" the distribution
    """
    # Step 1: Extract raw probability array (already provided as argument)
    raw_preds = raw_predictions
    
    # Step 2: Convert to log-space (logits)
    # Add epsilon to prevent log(0) errors
    epsilon = 1e-10
    logits = np.log(raw_preds + epsilon)
    
    # Step 3: Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Step 4: Numerical stability fix
    # Subtract max logit before exponential to prevent overflow/underflow
    shifted_logits = scaled_logits - np.max(scaled_logits)
    
    # Step 5: Re-apply softmax
    exp_values = np.exp(shifted_logits)
    calibrated_probs = exp_values / np.sum(exp_values)
    
    return calibrated_probs

def preprocess_image_for_model(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess EXACTLY as the test_flow / confusion-matrix pipeline:
    - Load with keras load_img(target_size=300,300) -> PIL resize (nearest/bilinear default)
    - img_to_array -> float32
    - Apply medical_preprocess (same as preprocessing_function in ImageDataGenerator)
    - Use ImageDataGenerator(preprocessing_function=medical_preprocess) to mirror internal flow
    - Return (preprocessed_with_batch, resized_rgb_array) where resized_rgb_array is the float32
      array prior to preprocessing (for overlay).
    """
    # Load and to array (float32)
    pil_img = keras_image.load_img(BytesIO(file_bytes), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(pil_img)  # float32

    # Mirror ImageDataGenerator behavior (preprocessing_function applied inside flow)
    datagen = ImageDataGenerator(preprocessing_function=medical_preprocess)
    img_batch = np.expand_dims(img_array, axis=0)
    processed_batch = next(datagen.flow(img_batch, batch_size=1, shuffle=False))

    # processed_batch[0] is the same tensor test_flow would feed to the model
    return processed_batch, img_array

def main():
    """Main application function with tabbed interface."""
    
    # Header Section
    st.markdown('<div class="main-header">🫁 Chest X-ray Assist: Explainable AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep Learning System for Multi-Disease Detection</div>', unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class="disclaimer">
    <strong>⚠️ Medical Disclaimer:</strong> This AI system is designed for research and educational purposes only.
    It is NOT approved for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar: Model Calibration Information
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        
        with st.expander("🎯 Temperature Scaling Calibration", expanded=False):
            st.markdown(f"""
            **Post-Hoc Calibration for Clinical Reliability**
            
            This model uses **Temperature Scaling** to improve prediction confidence calibration:
            
            **Calibration Temperature:** `T = {CALIBRATION_TEMPERATURE}`
            
            **Why Calibration Matters:**
            - Deep learning models often produce overconfident predictions
            - Raw softmax outputs may not reflect true probability
            - Temperature scaling improves Expected Calibration Error (ECE)
            
            **How It Works:**
            1. Obtain raw model predictions (logits)
            2. Scale logits by temperature: `logits / T`
            3. Apply softmax for calibrated probabilities
            
            **Clinical Impact:**
            - More reliable confidence scores for medical decision support
            - Better alignment between predicted confidence and actual accuracy
            - Improved trustworthiness in heterogeneous clinical settings
            
            **Reference:** Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
            
            ---
            
            *The displayed confidence scores are calibrated values, providing more clinically meaningful probability estimates than raw model outputs.*
            """)
        
        st.markdown("---")
        st.markdown("**Model:** EfficientNet-B3 with CLAHE")
        st.markdown("**Input Size:** 300×300 pixels")
        st.markdown("**Classes:** 4 lung conditions")
        st.markdown("**XAI Method:** Grad-CAM++")
    
    # Create tabs
    tabs = st.tabs(["🏥 X-Ray Analysis", "📊 Model Details & Performance", "📖 User Manual"])
    
    # Tab 1: X-Ray Analysis
    with tabs[0]:
        analysis_tab()
    
    # Tab 2: Model Details & Performance (Combined)
    with tabs[1]:
        model_details_and_performance_tab()
    
    # Tab 3: User Manual
    with tabs[2]:
        user_manual_tab()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <strong>CXR-Assistant v1.0 | Developed by Liew Jin Sze</strong><br>
        Data Science Project: Universiti Malaya<br><br>
        <strong>Powered by:</strong> EfficientNet-B3 • CLAHE Enhancement • Grad-CAM++ Explainability<br>
        <strong>Disclaimer:</strong> For Academic Research Purposes Only. Not for Clinical Diagnosis.
    </div>
    """, unsafe_allow_html=True)

def analysis_tab():
    """X-Ray analysis tab content."""
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Chest X-Ray")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a frontal chest X-ray in JPG or PNG format"
        )
        
        if uploaded_file is not None:
            try:
                # CRITICAL: Load and convert to RGB format (matches Keras ImageDataGenerator)
                # PIL.Image.open can load images in various formats (L, LA, RGB, RGBA, CMYK, etc.)
                # ImageDataGenerator always converts to RGB, so we must do the same
                image = Image.open(uploaded_file)
                
                # Force conversion to RGB mode (this is what Keras ImageDataGenerator does internally)
                # This ensures:
                # - Grayscale images (mode 'L') become RGB by replicating the channel
                # - RGBA images lose alpha channel
                # - Already RGB images remain unchanged
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to numpy array for display only
                image_array = np.array(image)
                file_bytes = uploaded_file.getvalue()
                
                st.image(image_array, caption="Uploaded X-Ray Image", use_container_width=True)
                
                # Enhanced Image info with metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📐 Width", f"{image.width} px")
                with col_b:
                    st.metric("📏 Height", f"{image.height} px")
                
                st.markdown("---")
                
                # Analysis button with enhanced description
                st.markdown("#### 🔬 Ready to Analyze")
                st.caption("Click below to start AI-powered analysis with Grad-CAM++ explainability")
                if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                    with st.spinner("🔄 Processing... Applying CLAHE enhancement and running deep learning inference..."):
                        analyze_image(file_bytes, image, col2)  # Pass raw bytes for preprocessing, PIL for display
                        
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                st.info("Please ensure the uploaded file is a valid image format.")
        else:
            st.info("👆 Please upload a chest X-ray image to begin analysis")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        st.info("Upload an image and click 'Analyze X-Ray' to see results")

def analyze_image(file_bytes: bytes, pil_image: Image.Image, display_column):
    """
    Analyze the uploaded X-ray image and display results.
    
    Args:
        file_bytes: Raw bytes of the uploaded image (for exact keras load)
        pil_image: PIL Image object in RGB mode (for display)
        display_column: Streamlit column to display results in
    """
    
    try:
        # Initialize classifier
        classifier = LungDiseaseClassifier()
        
        # Load model
        if not classifier.load_model():
            st.error("❌ Failed to load model. Please check the model file exists.")
            return
        
        # Preprocess image using Keras/PIL pipeline (matches provided deployment snippet)
        processed_image, resized_rgb = preprocess_image_for_model(file_bytes)
        
        # Step 1: Extract raw outputs from model.predict
        raw_output = classifier.model.predict(processed_image, verbose=0)
        
        # Handle potential list output
        if isinstance(raw_output, list):
            raw_output = raw_output[0]
        
        # Extract the raw probability array
        raw_predictions = raw_output[0]
        
        # Step 2-5: Apply Temperature Scaling for calibrated confidence scores
        # This improves Expected Calibration Error (ECE) for clinical reliability
        calibrated_predictions = apply_temperature_scaling(raw_predictions, CALIBRATION_TEMPERATURE)
        
        # Get all confidence scores (using calibrated predictions)
        class_labels = classifier.get_class_labels()
        
        # Create dictionaries for both raw and calibrated predictions (for audit)
        raw_confidence_dict = {label: float(score) for label, score in zip(class_labels, raw_predictions)}
        calibrated_confidence_dict = {label: float(score) for label, score in zip(class_labels, calibrated_predictions)}
        
        # Sort by calibrated confidence (descending)
        sorted_confidences = sorted(calibrated_confidence_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_raw_confidences = sorted(raw_confidence_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get top prediction
        predicted_class = sorted_confidences[0][0]
        raw_confidence_score = sorted_raw_confidences[0][1]
        calibrated_confidence_score = sorted_confidences[0][1]
        
        # Step 6: Display constraint - cap at 99.98% for UI if needed
        # This ensures clinical uncertainty is visible
        display_confidence = min(calibrated_confidence_score, 0.9998)
        
        # Display results in the provided column
        with display_column:
            # Top prediction card with enhanced animation
            st.markdown(f"""
            <div class="prediction-card animate-fade-in">
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">🏥 AI Diagnosis (Calibrated)</p>
                    <h1 style="margin: 15px 0; font-size: 2.8rem; text-transform: uppercase; letter-spacing: 2px;">{predicted_class}</h1>
                    <div style="background: rgba(255,255,255,0.2); border-radius: 50px; padding: 10px 20px; display: inline-block; margin-top: 10px;">
                        <h3 style="margin: 0; font-size: 1.5rem;">Calibrated Confidence: {display_confidence:.2%}</h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Internal audit display (raw vs calibrated)
            st.caption(f"🔬 **Internal Audit:** Raw Confidence: {raw_confidence_score:.4%} → Calibrated: {calibrated_confidence_score:.4%} (T={CALIBRATION_TEMPERATURE})")
            
            # All confidence scores
            st.markdown("### 📊 Detailed Confidence Scores (Calibrated)")
            
            # Create DataFrame for better visualization with display constraint
            sorted_confidences_display = [
                (disease, min(conf, 0.9998) if i == 0 else conf) 
                for i, (disease, conf) in enumerate(sorted_confidences)
            ]
            confidence_df = pd.DataFrame(sorted_confidences_display, columns=['Disease', 'Confidence'])
            confidence_df['Confidence'] = confidence_df['Confidence'].apply(lambda x: f"{x:.4%}")
            confidence_df['Rank'] = range(1, len(confidence_df) + 1)
            confidence_df = confidence_df[['Rank', 'Disease', 'Confidence']]
            
            # Display as styled table
            st.dataframe(
                confidence_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Disease": st.column_config.TextColumn("Disease", width="medium"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="medium"),
                }
            )
            
            # Progress bars for visual representation
            st.markdown("### 📈 Visual Confidence Distribution")
            for i, (disease, conf) in enumerate(sorted_confidences):
                st.markdown(f"**{disease.upper()}**")
                # Apply display constraint to top prediction
                display_conf = min(conf, 0.9998) if i == 0 else conf
                # Ensure value is between 0 and 1
                progress_value = min(max(float(display_conf), 0.0), 1.0)
                st.progress(progress_value)
                st.caption(f"{display_conf:.4%}")
            
            # Generate Grad-CAM visualization
            st.markdown("### 🔍 AI Explainability (Grad-CAM++)")
            
            try:
                with st.spinner("Generating visual explanation..."):
                    class_idx = class_labels.index(predicted_class)
                    original_display, overlay = generate_explanation_visualization(
                        classifier.model, processed_image, resized_rgb, class_idx
                    )
                    
                    # Display side by side
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.image(original_display, caption="Enhanced Image", use_container_width=True)
                    
                    with col_b:
                        st.image(overlay, caption="Grad-CAM++ Heatmap", use_container_width=True)
                    
                    # Interpretation
                    with st.expander("ℹ️ Understanding Grad-CAM++ Visualization"):
                        st.markdown(f"""
                        **Grad-CAM++ Explanation:**
                        - 🔴 **Red/Yellow regions**: Areas that strongly influenced the AI's decision
                        - 🔵 **Blue/Purple regions**: Areas with minimal influence on prediction
                        - The heatmap reveals which anatomical structures the AI focused on
                        
                        **Predicted Class:** {predicted_class} (Calibrated Confidence: {display_confidence:.2%})
                        
                        **Clinical Significance:**
                        This visualization provides transparency into the model's decision-making process,
                        helping medical professionals understand and validate AI predictions. The confidence
                        scores shown are calibrated using Temperature Scaling (T={CALIBRATION_TEMPERATURE}) to
                        provide more reliable probability estimates for clinical decision support.
                        """)
                        
            except Exception as e:
                st.warning(f"⚠️ Could not generate Grad-CAM visualization: {str(e)}")
            
            # Clinical recommendations
            st.markdown("### ⚕️ Clinical Recommendations")
            st.markdown("""
            <div class="metric-card">
            <strong>Important Reminders:</strong>
            <ul>
                <li>This is an AI-assisted preliminary analysis only</li>
                <li>Confirmation by radiologist and clinician is mandatory</li>
                <li>Consider patient history, symptoms, and additional tests</li>
                <li>Follow standard clinical protocols for diagnosis and treatment</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("Please check the image format and try again.")

def model_details_and_performance_tab():
    """Combined model details and performance metrics tab."""
    
    st.markdown("## 🏗️ Model Architecture & Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🧠 Base Architecture</h3>
        <p><strong>Model:</strong> EfficientNet-B3</p>
        <p><strong>Framework:</strong> TensorFlow / Keras</p>
        <p><strong>Input Size:</strong> 300 × 300 × 3</p>
        <p><strong>Output Classes:</strong> 4</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <h3>⚙️ Training Configuration</h3>
        <p><strong>Optimizer:</strong> Adam</p>
        <p><strong>Loss Function:</strong> Categorical Cross-Entropy</p>
        <p><strong>Batch Size:</strong> 32</p>
        <p><strong>Data Augmentation:</strong> Yes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🔬 Preprocessing Pipeline</h3>
        <p><strong>Enhancement:</strong> CLAHE (Contrast Limited AHE)</p>
        <p><strong>Color Space:</strong> LAB → RGB</p>
        <p><strong>Normalization:</strong> EfficientNet preprocessing</p>
        <p><strong>Clip Limit:</strong> 2.0</p>
        <p><strong>Tile Grid:</strong> 8 × 8</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Classification Classes</h3>
        <ol>
            <li><strong>COVID-19:</strong> Viral pneumonia patterns</li>
            <li><strong>Normal:</strong> Healthy lung tissue</li>
            <li><strong>Pneumonia:</strong> Bacterial lung infection</li>
            <li><strong>Tuberculosis:</strong> TB infection patterns</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance Metrics Section
    st.markdown("## 📊 Model Performance Evaluation")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Overall Accuracy", "98.0%", delta="Excellent")
    
    with col2:
        st.metric("📈 ROC-AUC Score", "0.9991", delta="Near Perfect")
    
    with col3:
        st.metric("📊 Macro F1-Score", "0.98", delta="Very High")
    
    with col4:
        st.metric("🔬 Test Samples", "2,716", delta="Robust Testing")
    
    st.markdown("---")
    
    # Per-class performance
    st.markdown("### 📋 Per-Class Performance Metrics")
    
    # Create performance DataFrame
    performance_data = {
        'Disease': ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis'],
        'Precision': [0.96, 0.99, 0.94, 1.00],
        'Recall': [0.97, 0.98, 0.98, 1.00],
        'F1-Score': [0.97, 0.98, 0.96, 1.00],
        'Support': [536, 1606, 202, 372]
    }
    
    df = pd.DataFrame(performance_data)
    
    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Disease": st.column_config.TextColumn("Disease", width="medium"),
            "Precision": st.column_config.ProgressColumn("Precision", format="%.2f", min_value=0, max_value=1),
            "Recall": st.column_config.ProgressColumn("Recall", format="%.2f", min_value=0, max_value=1),
            "F1-Score": st.column_config.ProgressColumn("F1-Score", format="%.2f", min_value=0, max_value=1),
            "Support": st.column_config.NumberColumn("Test Samples", format="%d"),
        }
    )
    
    # Detailed metrics explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Metric Definitions</h3>
        <p><strong>Precision:</strong> How many predicted positives are actually positive<br>
        <em>High precision = Few false positives</em></p>
        
        <p><strong>Recall (Sensitivity):</strong> How many actual positives were detected<br>
        <em>High recall = Few false negatives</em></p>
        
        <p><strong>F1-Score:</strong> Harmonic mean of precision and recall<br>
        <em>Balanced performance indicator</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Performance Highlights</h3>
        <ul>
            <li><strong>Tuberculosis:</strong> Perfect classification (100%)</li>
            <li><strong>Normal cases:</strong> 99% precision</li>
            <li><strong>COVID-19:</strong> 97% balanced performance</li>
            <li><strong>Pneumonia:</strong> 98% recall (excellent detection)</li>
        </ul>
        <p style="color: #16a34a; font-weight: bold; margin-top: 1rem;">
        ✅ Model shows excellent performance across all disease categories
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Explainable AI Section
    st.markdown("## 🔍 Explainable AI")
    st.markdown("""
    <div class="metric-card">
    <h3>Grad-CAM++ (Gradient-weighted Class Activation Mapping)</h3>
    <p>
    Our system uses <strong>Grad-CAM++</strong>, an advanced visualization technique that highlights
    the regions in the X-ray image that were most important for the model's decision. This provides:
    </p>
    <ul>
        <li><strong>Transparency:</strong> See what the AI "looks at" when making predictions</li>
        <li><strong>Validation:</strong> Verify that the model focuses on clinically relevant areas</li>
        <li><strong>Trust:</strong> Build confidence in AI-assisted diagnosis</li>
        <li><strong>Education:</strong> Learn from AI's pattern recognition</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model validation info
    st.markdown("### 🔬 Validation Protocol")
    st.markdown("""
    <div class="metric-card">
    <strong>Dataset Split:</strong>
    <ul>
        <li>Training Set: ~70% of total data with augmentation</li>
        <li>Validation Set: ~15% for hyperparameter tuning</li>
        <li>Test Set: ~15% (2,716 samples) for final evaluation</li>
    </ul>
    
    <strong>Cross-Validation:</strong> Stratified K-Fold to ensure balanced class representation
    
    <strong>Data Augmentation:</strong> Rotation (±15°), shifts, zoom, horizontal flip, brightness adjustment
    </div>
    """, unsafe_allow_html=True)

def user_manual_tab():
    """User manual and instructions."""
    
    st.markdown("## 📖 User Manual")
    
    st.markdown("""
    <div class="metric-card">
    <h3>🚀 Quick Start Guide</h3>
    
    <h4>Step 1: Prepare Your Image</h4>
    <ul>
        <li>✅ Use frontal chest X-ray images (PA or AP view)</li>
        <li>✅ Supported formats: JPG, JPEG, PNG</li>
        <li>✅ Image can be any size (will be resized automatically)</li>
        <li>✅ Both grayscale and color images are supported</li>
    </ul>
    
    <h4>Step 2: Upload and Analyze</h4>
    <ol>
        <li>Navigate to the <strong>"X-Ray Analysis"</strong> tab</li>
        <li>Click on the upload area and select your X-ray image</li>
        <li>Preview the uploaded image to ensure correct upload</li>
        <li>Click the <strong>"Analyze X-Ray"</strong> button</li>
        <li>Wait for the analysis to complete (usually 5-10 seconds)</li>
    </ol>
    
    <h4>Step 3: Interpret Results</h4>
    <ul>
        <li><strong>Top Prediction:</strong> The most likely diagnosis with confidence level</li>
        <li><strong>Confidence Scores:</strong> All possible diagnoses ranked by probability</li>
        <li><strong>Grad-CAM++ Visualization:</strong> Shows which areas influenced the decision</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="metric-card">
    <h3>⚠️ Important Guidelines</h3>
    
    <h4>Image Quality Requirements:</h4>
    <ul>
        <li>Use clear, properly exposed X-ray images</li>
        <li>Avoid heavily cropped or rotated images</li>
        <li>Ensure adequate contrast and brightness</li>
        <li>Remove any annotations or overlays if possible</li>
    </ul>
    
    <h4>Clinical Usage:</h4>
    <ul>
        <li>❌ DO NOT use for final clinical diagnosis</li>
        <li>✅ Use as a preliminary screening tool</li>
        <li>✅ Always confirm with qualified radiologists</li>
        <li>✅ Consider in context of patient history and symptoms</li>
        <li>✅ Follow standard medical protocols</li>
    </ul>
    
    <h4>Understanding Confidence Scores:</h4>
    <ul>
        <li><strong>90-100%:</strong> Very high confidence</li>
        <li><strong>70-89%:</strong> High confidence</li>
        <li><strong>50-69%:</strong> Moderate confidence</li>
        <li><strong>Below 50%:</strong> Low confidence - inconclusive</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="metric-card">
    <h3>🔧 Troubleshooting</h3>
    
    <h4>Common Issues:</h4>
    
    <strong>Problem:</strong> Upload fails or error message appears<br>
    <strong>Solution:</strong> Check file format (must be JPG/PNG), file size, and image integrity
    
    <strong>Problem:</strong> Analysis takes too long<br>
    <strong>Solution:</strong> Large images may take longer; try resizing to under 2000×2000 pixels
    
    <strong>Problem:</strong> Unexpected prediction results<br>
    <strong>Solution:</strong> Verify image quality, orientation, and that it's a chest X-ray
    
    <strong>Problem:</strong> Low confidence scores<br>
    <strong>Solution:</strong> Image may be ambiguous or of poor quality; consider retaking or using different image
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="metric-card">
    <h3>📞 Support & Information</h3>
    
    <h4>Technical Specifications:</h4>
    <ul>
        <li><strong>Model:</strong> EfficientNet-B3 with custom classification head</li>
        <li><strong>Training Data:</strong> Multi-source chest X-ray datasets</li>
        <li><strong>Preprocessing:</strong> CLAHE enhancement in LAB color space</li>
        <li><strong>Input Size:</strong> 300×300 pixels (automatic resizing)</li>
    </ul>
    
    <h4>Limitations:</h4>
    <ul>
        <li>Model trained on specific dataset distribution</li>
        <li>Performance may vary with different X-ray equipment</li>
        <li>Not validated for pediatric or pregnancy cases</li>
        <li>Cannot detect all possible lung pathologies</li>
    </ul>
    
    <h4>For Research Purposes:</h4>
    <p>This system is provided for educational and research purposes. If you're using this for 
    research, please ensure compliance with ethical guidelines and data privacy regulations 
    (HIPAA, GDPR, etc.).</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
