import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import joblib
import json
import os
from io import BytesIO
import base64

# Set page config for better mobile experience
st.set_page_config(
    page_title="Sports Celebrity Classifier",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load the model and class dictionary
@st.cache_resource
def load_model():
    try:
        model_path = 'model/best_model.pkl'
        if not os.path.exists(model_path):
            st.error("❌ Model file not found! Please ensure the model file exists in the model/ directory.")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_class_dict():
    try:
        with open('class_dictionary.json', 'r') as f:
            class_dict = json.load(f)
        # Create reverse mapping from index to class name with proper formatting
        return {v: k.replace('_', ' ').title() for k, v in class_dict.items()}
    except FileNotFoundError:
        st.error("❌ Class dictionary file not found! Please ensure class_dictionary.json exists in the project directory.")
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format in class dictionary file!")
    except Exception as e:
        st.error(f"❌ Error loading class dictionary: {str(e)}")
    return {}

def preprocess_image(image):
    """
    Preprocess the image for model prediction
    Converts to grayscale, resizes, and normalizes the image
    """
    try:
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            # Convert to grayscale
            image = ImageOps.grayscale(image)
            img_array = np.array(image)
        else:
            # If it's already an array, ensure it's in the right format
            if len(image.shape) == 3:
                img_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img_array = image
        
        # Resize to 100x100 (adjust based on your model's expected input)
        img_resized = cv2.resize(img_array, (100, 100))
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized / 255.0
        
        # Flatten the image to 1D array (100*100 = 10000 features)
        img_flattened = img_normalized.reshape(1, -1)
        
        return img_flattened
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        st.error(f"Image shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")
        return None

def get_image_download_link(img, filename="image.png", text="Download sample image"):
    """Generate a download link for the sample image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        max-width: 800px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: 1px solid #45a049;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏆 Sports Celebrity Classifier")
    st.markdown("### Upload an image of a sports celebrity and our AI will identify who it is!")
    
    # Load model and class dictionary with loading states
    with st.spinner("🔍 Loading the AI model..."):
        model = load_model()
        if model is None:
            st.error("Failed to load the AI model. Please check the error message above.")
            return
            
        idx_to_class = load_class_dict()
        if not idx_to_class:
            st.error("Failed to load class labels. Please ensure class_dictionary.json exists and is valid.")
            return
    
    # Add a sample image section for testing
    st.markdown("### 🧪 Try with a sample image")
    sample_options = ["Select a sample", "Roger Federer", "Lionel Messi", "Serena Williams"]
    sample_choice = st.selectbox("Choose a sample image to test:", sample_options, index=0)
    
    uploaded_file = None
    
    # Handle sample image selection
    if sample_choice != "Select a sample":
        sample_name = sample_choice.lower().replace(' ', '_')
        sample_path = f"samples/{sample_name}.jpg"
        try:
            sample_img = Image.open(sample_path)
            st.image(sample_img, caption=f"Sample: {sample_choice}", use_column_width=True)
            
            # Convert PIL Image to file-like object for consistency with file_uploader
            img_byte_arr = BytesIO()
            sample_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            uploaded_file = BytesIO(img_byte_arr.read())
            uploaded_file.name = f"sample_{sample_name}.png"
        except Exception as e:
            st.warning(f"Could not load sample image: {str(e)}")
    
    # File uploader
    st.markdown("### 📤 Or upload your own image")
    file_upload = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Upload an image of a sports celebrity",
        key="file_uploader"
    )
    
    # Use either the uploaded file or the sample file
    image_file = file_upload if file_upload is not None else uploaded_file
    
    if image_file is not None:
        try:
            # Display the uploaded image with a nice border
            st.markdown("### 📷 Your Image")
            col1, col2 = st.columns([1, 3])
            with col1:
                image = Image.open(image_file)
                st.image(
                    image, 
                    caption="Uploaded Image", 
                    use_column_width=True,
                    output_format="PNG"
                )
            
            # Add image info
            with col2:
                st.markdown("### ℹ️ Image Details")
                st.write(f"**Format:** {image.format if hasattr(image, 'format') else 'N/A'}")
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Mode:** {image.mode}")
            
            # Add a separator
            st.markdown("---")
            
            # Preprocess and predict
            with st.spinner('🔍 Analyzing the image...'):
                processed_img = preprocess_image(image)
                
                if processed_img is not None:
                    # Get prediction
                    try:
                        prediction = model.predict(processed_img)
                        probabilities = model.predict_proba(processed_img)[0]
                        
                        # Get top 3 predictions
                        top3_indices = np.argsort(probabilities)[-3:][::-1]
                        top3_classes = [idx_to_class[i] for i in top3_indices if i in idx_to_class]
                        top3_probs = [probabilities[i] for i in top3_indices if i in idx_to_class]
                        
                        # Display results
                        st.markdown("## 🎯 Prediction Results")
                        
                        # Show top prediction with confidence
                        if top3_classes and top3_probs:
                            st.markdown(
                                f"<div class='prediction-card'>"
                                f"<h3>🎉 Predicted: <strong>{top3_classes[0]}</strong></h3>"
                                f"<div>Confidence: <strong>{top3_probs[0]*100:.1f}%</strong></div>"
                                f"<div class='confidence-bar'><div class='confidence-fill' style='width: {top3_probs[0]*100}%'></div></div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Show top 3 predictions in a nice way
                            st.markdown("### 📊 Top Predictions")
                            for i, (cls, prob) in enumerate(zip(top3_classes, top3_probs), 1):
                                st.markdown(
                                    f"<div style='margin: 0.5rem 0;'>"
                                    f"<div style='display: flex; justify-content: space-between; margin-bottom: 0.2rem;'>"
                                    f"<span>{i}. {cls}</span>"
                                    f"<span><strong>{prob*100:.1f}%</strong></span>"
                                    f"</div>"
                                    f"<div class='confidence-bar'><div class='confidence-fill' style='width: {prob*100}%'></div></div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.warning("No valid predictions were returned by the model.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.error("This might be due to a model compatibility issue. Please try a different image.")
        
        except Exception as e:
            st.error(f"❌ An error occurred while processing the image: {str(e)}")
            st.error("Please try with a different image or check the console for more details.")
            st.stop()
    
    # Add some instructions and information
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        1. **Upload an image** of a sports celebrity using the file uploader above
        2. Or **select a sample image** from the dropdown menu to test the app
        3. The AI will analyze the image and show you the top predictions
        4. Results include confidence levels for each prediction
        
        *Note: The model has an accuracy of 81.25% on the test dataset.*
        """)
    
    # Supported celebrities
    st.markdown("### 🏆 Supported Sports Celebrities")
    celebs = sorted([name.replace('_', ' ').title() for name in idx_to_class.values()])
    st.markdown(" • ".join(celebs))
    
    # Add some tips for best results
    with st.expander("💡 Tips for better results", expanded=False):
        st.markdown("""
        - Use clear, well-lit photos of the celebrity's face
        - Front-facing portraits work best
        - Avoid group photos or images with multiple faces
        - The model works best with images where the face is clearly visible
        - Supported formats: JPG, JPEG, PNG
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
        <p>This is a demo application for educational purposes only.</p>
        <p>The predictions are based on a machine learning model with 81.25% test accuracy and may not always be correct.</p>
        <p>Model accuracy: 0.8125 (81.25%) on test dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
