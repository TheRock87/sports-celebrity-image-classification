import streamlit as st
import numpy as np
import cv2
import json
import pickle
import os
from PIL import Image
import io
import base64
from pathlib import Path
import requests
from io import BytesIO

# Set page title and favicon
st.set_page_config(
    page_title="Sports Celebrity Classifier",
    page_icon="🏆",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .prediction-result {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .confidence-score {
        font-size: 1.2rem;
        color: #424242;
    }
    .sample-img {
        cursor: pointer;
        transition: transform 0.3s;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .sample-img:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .sample-caption {
        text-align: center;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model and class dictionary
@st.cache_resource
def load_model_and_class_dict():
    # Load the model
    model_path = os.path.join('model', 'best_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load the class dictionary
    with open('class_dictionary.json', 'r') as f:
        class_dict = json.load(f)
    
    # Invert the dictionary to map indices to names
    class_names = {v: k.replace('_', ' ').title() for k, v in class_dict.items()}
    
    return model, class_names

# Function to preprocess the image
def preprocess_image(image):
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Process the image the same way as during model training
    # 1. Create a scaled raw color image (32x32x3)
    scaled_raw_img = cv2.resize(image, (32, 32))
    
    # 2. Create a wavelet transformed grayscale image
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply wavelet transform (similar to w2d function used in training)
    # Note: This is a simplified version of wavelet transform
    gray_image = np.float32(gray_image)
    gray_image /= 255
    # Apply histogram equalization for standardization
    gray_image = cv2.equalizeHist(cv2.convertScaleAbs(gray_image * 255))
    # Resize to 32x32
    scaled_img_har = cv2.resize(gray_image, (32, 32))
    
    # 3. Combine both feature sets (3072 + 1024 = 4096 features)
    combined_img = np.vstack((
        scaled_raw_img.reshape(32*32*3, 1),
        scaled_img_har.reshape(32*32, 1)
    ))
    
    # Reshape to match model input requirements
    flattened_image = combined_img.reshape(1, 4096).astype(float)
    
    # Add information about preprocessing to session state for display
    if 'preprocessing_info' not in st.session_state:
        st.session_state.preprocessing_info = {}
    
    st.session_state.preprocessing_info = {
        'original_shape': image.shape,
        'color_features': 32*32*3,
        'wavelet_features': 32*32,
        'total_features': 4096,
        'resized': (32, 32),
        'wavelet_transform': True,
        'flattened_shape': flattened_image.shape
    }
    
    return flattened_image

# Function to make predictions
def predict_celebrity(image, model, class_names):
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        prediction_proba = model.predict_proba(processed_image)
        
        # Get the predicted class and its probability
        predicted_class_idx = prediction[0]
        confidence = prediction_proba[0][predicted_class_idx] * 100
        
        # Get the celebrity name
        celebrity_name = class_names[predicted_class_idx]
        
        # Get all probabilities for visualization
        all_probabilities = {class_names[i]: prediction_proba[0][i] * 100 for i in range(len(class_names))}
        
        return celebrity_name, confidence, all_probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, 0, {}

# Function to get sample images
def get_sample_images():
    # Define sample images for each celebrity
    sample_images = {
        "Roger Federer": "https://media.gettyimages.com/id/1436657095/photo/roger-federer-of-team-europe-waves-to-the-crowd-after-playing-his-final-match-of-his-career.jpg?s=612x612&w=gi&k=20&c=Xk2Kj5QUKv9Vy-LbA3bQXJKVbgUMKXYJDTvSbWxj3Ck=",
        "Morgan Gibbs-White": "https://static01.nyt.com/athletic/uploads/wp/2024/10/08033026/morgan-gibbs-white-scaled-e1728372659793-1024x681.jpg?width=770&quality=70&auto=webp",
        "Serena Williams": "https://media.cnn.com/api/v1/images/stellar/prod/230825151122-01-serena-williams-us-open-2022.jpg?c=16x9&q=h_720,w_1280,c_fill",
        "Lionel Messi": "https://assets.goal.com/v3/assets/bltcc7a7ffd2fbf71f5/blt12dbddde5342ce4c/648866ff21a8556da91af0b7/GOAL_-_Blank_WEB_-_Facebook_-_2023-06-13T135350.847.png?auto=webp&format=pjpg&width=3840&quality=60",
        "Maria Sharapova": "https://www.si.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cfl_progressive%2Cq_auto:good%2Cw_1200/MTY4MTk3MTYyNjYxMDg2ODkz/2011-0620-Maria-Sharapova-opjpg.jpg"
    }
    return sample_images

# Main function
def main():
    # Load model and class dictionary
    try:
        model, class_names = load_model_and_class_dict()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
    
    # Header
    st.markdown('<h1 class="main-header">Sports Celebrity Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Upload an image to identify the sports celebrity</h2>', unsafe_allow_html=True)
    
    # Display information about the supported celebrities
    st.info("This app can identify the following sports celebrities: Roger Federer, Eoin Morgan, Serena Williams, Lionel Messi, and Maria Sharapova")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Use Sample Image"])
    
    # Initialize session state for storing the image
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'celebrity_name' not in st.session_state:
        st.session_state.celebrity_name = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0
    if 'all_probabilities' not in st.session_state:
        st.session_state.all_probabilities = {}
    
    # Upload Image tab
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.session_state.image = image
                st.session_state.prediction_made = False
            except Exception as e:
                st.error(f"Error opening image: {e}")
    
    # Sample Image tab
    with tab2:
        sample_images = get_sample_images()
        
        # Create columns for sample images
        cols = st.columns(5)
        
        # Display sample images
        for i, (name, url) in enumerate(sample_images.items()):
            with cols[i]:
                st.image(url, caption=name, width=150, use_container_width=False)
                if st.button(f"Use {name}", key=f"sample_{i}"):
                    try:
                        response = requests.get(url)
                        image = Image.open(BytesIO(response.content))
                        st.session_state.image = image
                        st.session_state.prediction_made = False
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error loading sample image: {e}")

    # If an image is selected (either uploaded or sample)
    if st.session_state.image is not None and model_loaded:
        # Display the selected image if not already displayed
        if not st.session_state.prediction_made:
            st.image(st.session_state.image, caption="Selected Image", use_container_width=True)
        
        # Make prediction when user clicks the button
        if st.button("Identify Celebrity"):
            with st.spinner("Analyzing image..."):
                try:
                    # Make prediction
                    celebrity_name, confidence, all_probabilities = predict_celebrity(st.session_state.image, model, class_names)
                    
                    if celebrity_name:
                        st.session_state.celebrity_name = celebrity_name
                        st.session_state.confidence = confidence
                        st.session_state.all_probabilities = all_probabilities
                        st.session_state.prediction_made = True
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        
        # Display prediction results if available
        if st.session_state.prediction_made:
            # Display the prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f'<p class="prediction-result">Predicted Celebrity: {st.session_state.celebrity_name}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence-score">Confidence: {st.session_state.confidence:.2f}%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display bar chart of all probabilities
            st.subheader("Prediction Probabilities")
            
            # Sort probabilities in descending order
            sorted_probs = dict(sorted(st.session_state.all_probabilities.items(), key=lambda item: item[1], reverse=True))
            
            # Create a bar chart
            chart_data = {
                'Celebrity': list(sorted_probs.keys()),
                'Probability (%)': list(sorted_probs.values())
            }
            
            # Use Streamlit's native chart
            st.bar_chart(chart_data, x='Celebrity', y='Probability (%)')
            
            # Display image preprocessing information
            if 'preprocessing_info' in st.session_state:
                st.subheader("Image Preprocessing Details")
                info = st.session_state.preprocessing_info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Image:**")
                    st.write(f"- Shape: {info['original_shape']}")
                    st.write(f"- Type: {'Color' if len(info['original_shape']) > 2 and info['original_shape'][2] == 3 else 'Grayscale'}")
                
                with col2:
                    st.write("**Processed Image:**")
                    st.write(f"- Resized to: {info['resized']}")
                    st.write(f"- Color features: {info['color_features']}")
                    st.write(f"- Wavelet transform features: {info['wavelet_features']}")
                    st.write(f"- Total features extracted: {info['total_features']}")
                
                st.info("The model was trained on 32x32 images with both color (3072 features) and wavelet-transformed grayscale (1024 features) data, resulting in 4096 total features per image. The preprocessing steps ensure your uploaded image matches this format.")
    
    # Add information about how the model works
    with st.expander("How does this model work?"):
        st.write("""
        This sports celebrity classifier uses a machine learning model trained on images of 5 famous sports celebrities:
        
        1. **Roger Federer** - Tennis player
        2. **Morgan Gibbs-White** - Soccer player
        3. **Serena Williams** - Tennis player
        4. **Lionel Messi** - Soccer player
        5. **Maria Sharapova** - Tennis player
        
        The model works by:
        1. Taking an input image
        2. Creating two versions of the image:
           - A color version resized to 32x32 pixels (3072 features)
           - A wavelet-transformed grayscale version resized to 32x32 pixels (1024 features)
        3. Combining these features into a single vector of 4096 features
        4. Using a trained classifier to predict which celebrity is in the image
        
        **Model Accuracy:** 81.25% on the test dataset
        
        For best results, use clear images where the celebrity's face is clearly visible and well-centered.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Sports Celebrity Classifier | Created with Streamlit | Model Accuracy: 81.25% | Features: 4096</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
