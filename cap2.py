import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from PIL import Image, ImageDraw, ImageFont
import time
import threading
import winsound  # For sound alerts on Windows

# Initialize the camera and load the animal classification model
cap = cv2.VideoCapture(0)
model = load_model(r"E:\cap s2\model\animal_classifier_mobilenet.h5")

# Define animal labels
labels = ["كلب", "صقر", "بومه", "ثعبان"]

# Set image processing parameters
imgsize = 224  # Ensure this matches the model's input size

# Load Arabic font
font_path = r"F:\IBMPlexSansArabic-Thin.ttf"
font_size = 60
font = ImageFont.truetype(font_path, font_size)

# Confidence threshold
confidence_threshold = 92

# Variables for tracking the last detected label and alerting
last_detected_label = ""
display_start_time = None
alert_active = False

# Function to start a continuous alert
def continuous_alert():
    global alert_active
    while alert_active:
        winsound.Beep(1000, 500)  # Beep sound (frequency, duration in ms)

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        continue

    img_output = frame.copy()

    # Prepare frame for the classifier
    img_resized = cv2.resize(frame, (imgsize, imgsize))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the animal in the frame
    predictions = model.predict(img_array)
    detected_label = labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Check if the detected label has high confidence
    if confidence > confidence_threshold and detected_label in labels:
        reshaped_text = reshape(detected_label)
        bidi_text = get_display(reshaped_text)

        # Start the display timer if it's a new label
        if detected_label != last_detected_label:
            display_start_time = time.time()
            last_detected_label = detected_label
            alert_active = False  # Reset alert

        # If the label has been displayed for 3 seconds, start the alert
        elif time.time() - display_start_time > 3 and not alert_active:
            alert_active = True
            alert_thread = threading.Thread(target=continuous_alert)
            alert_thread.start()

        # Drawing the Arabic text using Pillow
        img_pil = Image.fromarray(img_output)
        draw = ImageDraw.Draw(img_pil)
        text_size = draw.textbbox((0, 0), bidi_text, font=font)
        text_position = ((frame.shape[1] - text_size[2]) // 2, 50)  # Center text on the screen
        draw.text(text_position, bidi_text, font=font, fill=(255, 255, 255))
        img_output = np.array(img_pil)
    else:
        # Reset if no animal is detected with high confidence
        last_detected_label = ""
        display_start_time = None
        alert_active = False

    # Show the output in OpenCV window
    cv2.imshow("Animal Detection", img_output)

    # Press 'q' to exit the display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
