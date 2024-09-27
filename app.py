import os
import sys
import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for development and PyInstaller. """
    try:
        # PyInstaller creates a temporary folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Load the model
model_json_path = resource_path("Facial Expression Recognition.json")
model_weights_path = resource_path("fer.h5")

with open(model_json_path, "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights(model_weights_path)

# Load Haar Cascade for face detection
face_haar_cascade_path = resource_path('haarcascade_frontalface_default.xml')
face_haar_cascade = cv2.CascadeClassifier(face_haar_cascade_path)

# Start video capture
cap = cv2.VideoCapture(0)

# Create a window with the desired name
cv2.namedWindow('Facial emotion analysis', cv2.WINDOW_NORMAL)  # Create a normal window

# Get screen width and height
screen_width = 1920  # Set this to your actual screen width
screen_height = 1080  # Set this to your actual screen height

# Resize window to fill the screen (maximized)
cv2.resizeWindow('Facial emotion analysis', screen_width, screen_height)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    # Check if any faces are detected
    if len(faces_detected) > 0:
        # Process only the first detected face
        x, y, w, h = faces_detected[0]  # Get the first face's coordinates

        # Draw rectangle around the face
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

        # Crop the region of interest (face area) from the image
        roi_gray = gray_img[y:y + h, x:x + w]  # corrected to use h for height
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Make predictions
        predictions = model.predict(img_pixels)

        # Find the index of the highest probability
        max_index = np.argmax(predictions[0])

        # Define emotions and get the predicted emotion
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        # Display the predicted emotion on the image
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image in the resized window
    cv2.imshow('Facial emotion analysis', test_img)

    # Check for key presses: 'q' to quit or 'Esc' to exit
    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:  # 27 is the ASCII code for Esc
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
