import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

# --- App Configuration & Model Loading ---
# Set the title and a brief description for the Streamlit app
st.title("Building Segmentation using Satellite Imagery")
st.write("Upload a satellite image to see buildings segmented and highlighted.")

# Define the expected input size for the model
IMG_SIZE = (256, 256)

# The model file 'inria_unet.h5' needs to be in the same directory as this script
# Use st.cache_resource to load the model only once, which speeds up the app.
@st.cache_resource
def load_segmentation_model():
    """
    Loads the pre-trained Keras model for segmentation.
    This function is cached to prevent reloading the model on every rerun.
    """
    try:
        model_path = "inria_unet.h5"
        # The model needs to be in the same directory as app.py for Streamlit to find it.
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure 'inria_unet.h5' is in the root directory of your app.")
        return None

# Load the model
model = load_segmentation_model()

# --- Image Processing Functions ---

def preprocess_image(image_data):
    """
    Resizes an image to the model's input size and normalizes it.
    Args:
        image_data (np.ndarray): The image as a NumPy array.
    Returns:
        np.ndarray: The preprocessed image as a numpy array, or None on error.
    """
    try:
        # Resize the image to the required dimensions for the model
        img_resized = cv2.resize(image_data, IMG_SIZE)
        # Convert to float and normalize pixel values to be between 0 and 1
        img_normalized = img_resized.astype("float32") / 255.0
        # Add a batch dimension to the image (e.g., from (256, 256, 3) to (1, 256, 256, 3))
        img_preprocessed = np.expand_dims(img_normalized, axis=0)
        
        return img_preprocessed
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def postprocess_mask(prediction, original_size):
    """
    Processes the model's prediction to create a clean binary mask.
    Args:
        prediction (np.ndarray): The raw output from the model.
        original_size (tuple): The size of the original image (width, height).
    Returns:
        np.ndarray: A binary mask resized back to the original image's size.
    """
    # The prediction is a floating-point value. We apply a threshold to get a binary mask.
    mask = (prediction > 0.5).astype("uint8")
    # Squeeze the array to remove extra dimensions, then resize back to the original image size
    mask_resized = cv2.resize(mask[0, :, :, 0], original_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized

def create_masked_overlay(original_img, mask):
    """
    Creates a visual overlay of the mask on the original image.
    Args:
        original_img (np.ndarray): The original image array.
        mask (np.ndarray): The binary mask array.
    Returns:
        np.ndarray: An image with the buildings highlighted.
    """
    # Create a red overlay for the buildings
    red_overlay = np.zeros_like(original_img, dtype=np.uint8)
    red_overlay[:, :, 2] = 255  # Set the red channel to max
    
    # Use the mask to blend the original image and the red overlay
    # The mask acts as a weight. Buildings will be a blend of original and red.
    overlay_img = cv2.addWeighted(original_img, 1.0, red_overlay, 0.4, 0, dtype=cv2.CV_8U)
    overlay_img[mask == 0] = original_img[mask == 0]
    
    return overlay_img

# --- Streamlit UI and Logic ---

# Use st.file_uploader to get an image file from the user
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # If a file is uploaded, convert it to an OpenCV image array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create two columns for a side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(original_img, caption="Original Satellite Image", use_column_width=True)
        
    # Add a button to trigger the segmentation process
    if st.button("Segment Buildings"):
        if model is not None:
            # Show a spinner while processing
            with st.spinner('Processing image and running model...'):
                original_size = (original_img.shape[1], original_img.shape[0])
                
                # Preprocess and predict
                preprocessed_img = preprocess_image(original_img)
                if preprocessed_img is not None:
                    prediction = model.predict(preprocessed_img)
                    
                    # Post-process the prediction
                    mask = postprocess_mask(prediction, original_size)
                    
                    # Create the final overlay image
                    overlay_img = create_masked_overlay(original_img, mask)
                    
                    # Display the segmented image in the second column
                    with col2:
                        st.header("Segmented Buildings")
                        st.image(overlay_img, caption="Buildings Highlighted", use_column_width=True)
        else:
            st.warning("Model is not loaded. Please check the `inria_unet.h5` file.")


