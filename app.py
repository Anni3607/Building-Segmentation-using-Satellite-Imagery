
import os
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# --- App Configuration ---
app = Flask(__name__)
# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths for temporary uploads and results
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
# Create the directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
# Set the upload and results folders for the Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# --- Model Loading ---
MODEL_PATH = os.path.join(BASE_DIR, 'inria_unet.h5')
# Define the expected input size for the model
IMG_SIZE = (256, 256) 
# Load the pre-trained model. We use try-except to handle potential errors.
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

def preprocess_image(image_path):
    """
    Loads an image, resizes it to the model's input size, and normalizes it.
    Args:
        image_path (str): The file path of the image.
    Returns:
        np.ndarray: The preprocessed image as a numpy array, or None on error.
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        # Check if the image was read correctly
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Resize the image to the required dimensions for the model
        img_resized = cv2.resize(img, IMG_SIZE)
        # Convert to float and normalize pixel values to be between 0 and 1
        img_normalized = img_resized.astype("float32") / 255.0
        # Add a batch dimension to the image (e.g., from (256, 256, 3) to (1, 256, 256, 3))
        img_preprocessed = np.expand_dims(img_normalized, axis=0)
        
        return img_preprocessed
    except Exception as e:
        print(f"Error preprocessing image: {e}")
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
    # The prediction is usually a floating-point value. We apply a threshold to get a binary mask.
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
    # Resize the mask to match the original image size for overlaying
    mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Create a red overlay for the buildings
    red_overlay = np.zeros_like(original_img, dtype=np.uint8)
    red_overlay[:, :, 2] = 255  # Set the red channel to max
    
    # Use the mask to blend the original image and the red overlay
    # The mask acts as an alpha channel. Buildings will be red, other areas will be original image.
    overlay_img = cv2.addWeighted(original_img, 1.0, red_overlay, 0.4, 0, dtype=cv2.CV_8U)
    overlay_img[mask_resized == 0] = original_img[mask_resized == 0]
    
    return overlay_img

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route to handle image uploads and display segmentation results.
    """
    if request.method == "POST":
        # Check if a file was uploaded
        if 'image_file' not in request.files:
            return render_template("index.html", error="No file part")
        
        file = request.files['image_file']
        
        if file.filename == '':
            return render_template("index.html", error="No selected file")

        if file:
            # Generate a unique filename to prevent conflicts
            filename = str(hash(file.filename + str(np.random.rand()))) + ".jpg"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if model is None:
                return render_template("index.html", error="Model not loaded. Please check model path.")

            # Get the original image to keep its size
            original_img = cv2.imread(file_path)
            original_size = (original_img.shape[1], original_img.shape[0])

            # Preprocess the image for the model
            preprocessed_img = preprocess_image(file_path)
            if preprocessed_img is None:
                return render_template("index.html", error="Failed to preprocess image.")

            # Make a prediction
            prediction = model.predict(preprocessed_img)
            
            # Post-process the prediction to get the final mask
            mask = postprocess_mask(prediction, original_size)
            
            # Save the original image and the masked overlay for display
            original_img_path = os.path.join('uploads', filename)
            
            # Create a visual overlay of the mask on the original image
            overlay_img = create_masked_overlay(original_img, mask)
            
            # Save the result
            result_filename = "result_" + filename
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, overlay_img)
            result_path_relative = os.path.join('results', result_filename)

            return render_template(
                "index.html",
                original_image=original_img_path,
                result_image=result_path_relative
            )

    # For GET requests, just render the main page
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
