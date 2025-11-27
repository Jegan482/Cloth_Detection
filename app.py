
### **2. The Flask Application (`app.py`)**


import os
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Define paths for uploading and storing results
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure the upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Load YOLO model ---
# IMPORTANT: Provide the correct path to your model weights file.
# Update this path to where your 'best.pt' file is located.
MODEL_PATH = "best.pt" # <<< CHANGE THIS PATH

# Set a confidence threshold for detections
CONF_THRESHOLD = 0.25

# Load the model once when the application starts
try:
    model = YOLO(MODEL_PATH)
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    # If the model fails to load, the app can still run but will show an error on processing.
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page which includes the file upload form.
    Processes the uploaded image on POST request.
    """
    if request.method == 'POST':
        # Check if the model was loaded successfully
        if not model:
            return render_template('index.html', error="Model is not loaded. Please check the server logs.")

        # Check if a file was uploaded in the request
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return render_template('index.html', error="No file selected for uploading.")

        if file:
            try:
                # Save the uploaded file temporarily
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logging.info(f"File '{filename}' uploaded successfully.")

                # --- Run inference ---
                # Open the image with PIL
                img_pil = Image.open(filepath).convert("RGB")
                
                # Convert PIL Image to an OpenCV-compatible format (NumPy array)
                img = np.array(img_pil)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

                # Run YOLO prediction on the image
                results = model.predict(source=img, conf=CONF_THRESHOLD, verbose=False)
                logging.info(f"Inference completed for '{filename}'.")

                # --- Draw bounding boxes on the image ---
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = f"{model.names[cls_id]} {conf:.2f}"
                        
                        # Draw rectangle and text
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    logging.info(f"Drew {len(results[0].boxes)} bounding boxes.")
                else:
                    logging.info("No objects detected.")

                # Save the processed image to the results folder
                result_filename = f"result_{filename}"
                result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                cv2.imwrite(result_filepath, img)
                logging.info(f"Result image saved to '{result_filepath}'.")

                # Render the result page with the path to the processed image
                return render_template('results.html', image_name=result_filename)
            
            except Exception as e:
                logging.error(f"An error occurred during processing: {e}")
                return render_template('index.html', error="An error occurred while processing the image.")

    # For GET requests, just display the upload form
    return render_template('index.html')

@app.route('/results/<filename>')
def send_result_image(filename):
    """
    Serves the processed image from the results directory.
    """
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    # Runs the Flask app
    # host='0.0.0.0' makes the app accessible from your local network
    app.run(host='0.0.0.0', port=5000, debug=True)