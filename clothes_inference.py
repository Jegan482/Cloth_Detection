# Import necessary libraries
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Set the path to your trained YOLO model weights.
# This path should correspond to where your model file is located in your Kaggle input directory.
MODEL_PATH = "best.pt"
# Provide a URL for the image you want to perform inference on.
IMAGE_URL = "https://images.unsplash.com/photo-1630412990381-fbbbcf86aa42?q=80&w=672&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# Set the confidence threshold for detections. Objects with a confidence score below this value will be ignored.
CONF_THRESHOLD = 0.25

# --- Load the YOLO model ---
# Initialize the YOLO model from the specified weights file.
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit or handle the error appropriately
    exit()

# --- Image loading and preparation ---
# Download the image from the provided URL.
try:
    response = requests.get(IMAGE_URL)
    response.raise_for_status()  # This will raise an HTTPError if the download failed
    # Open the image from the downloaded content and convert it to RGB format.
    img_pil = Image.open(BytesIO(response.content)).convert("RGB")
except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
    # Exit or handle the error appropriately
    exit()


# Convert the PIL (Pillow) Image to an OpenCV compatible format (NumPy array).
# It's important to ensure the data type is uint8, which is standard for images.
img = np.array(img_pil, dtype=np.uint8)

# OpenCV uses the BGR color format by default, while PIL uses RGB.
# We need to convert from RGB to BGR for processing with OpenCV.
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# --- Run YOLO inference ---
# Use the model to predict objects in the source image.
# 'conf' sets the confidence threshold for detections.
# 'verbose=False' prevents detailed prediction information from being printed to the console.
results = model.predict(source=img, conf=CONF_THRESHOLD, verbose=False)

# --- Process and display results ---
# Check if any detections were made.
if results[0].boxes is not None and len(results[0].boxes) > 0:
    # Iterate through each detected bounding box.
    for box in results[0].boxes:
        # Extract the coordinates of the bounding box (x-start, y-start, x-end, y-end).
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Get the confidence score of the detection.
        conf = float(box.conf[0])
        # Get the class ID of the detected object.
        cls_id = int(box.cls[0])
        # Create a label with the class name and the confidence score.
        label = f"{model.names[cls_id]} {conf:.2f}"

        # Draw a rectangle around the detected object.
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put the label text above the bounding box.
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print("⚠️ No detections found!")

# --- Display the final image ---
# Convert the image back from BGR (OpenCV's format) to RGB for correct color display with Matplotlib.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a figure to display the image.
plt.figure(figsize=(10, 8))
# Show the image.
plt.imshow(img_rgb)
# Turn off the axes for a cleaner look.
plt.axis("off")
# Display the plot.
plt.show()