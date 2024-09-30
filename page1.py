import tkinter as tk
import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk, ImageDraw, ImageFont  # Pillow modules for handling images
import os
import sys

class FacialExpressionDetection(tk.Frame):
    def __init__(self, parent, switch_page_callback):
        super().__init__(parent)
        self.switch_page_callback = switch_page_callback

        # Add button to switch pages
        switch_button = tk.Button(self, text="Switch Page", command=self.switch_to_second_page)
        switch_button.pack(side=tk.TOP, anchor='ne')

        # Label to display the video feed, set to expand and fill the window
        self.label = tk.Label(self)
        self.label.pack(fill=tk.BOTH, expand=True)

        # Load resources
        self.model = self.load_model()
        self.face_haar_cascade = self.load_cascade()
        self.font = self.load_font()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.show_frame()

    def resource_path(self, relative_path):
        """ Get the absolute path to the resource, works for development and PyInstaller. """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def load_model(self):
        """ Load the facial expression model """
        model_json_path = self.resource_path("Facial Expression Recognition.json")
        model_weights_path = self.resource_path("fer.h5")

        with open(model_json_path, "r") as json_file:
            model = model_from_json(json_file.read())
        model.load_weights(model_weights_path)
        return model

    def load_cascade(self):
        """ Load Haar Cascade for face detection """
        face_haar_cascade_path = self.resource_path('haarcascade_frontalface_default.xml')
        return cv2.CascadeClassifier(face_haar_cascade_path)

    def load_font(self):
        """ Load the font for emoji and text rendering """
        font_path = self.resource_path("DejaVuSans.ttf")
        return ImageFont.truetype(font_path, 32)

    def show_frame(self):
        """ Captures and displays the real-time video feed with facial emotion detection """
        ret, test_img = self.cap.read()  # Capture frame
        if not ret:
            return

        # Get the size of the label (which fills the window)
        label_width = self.label.winfo_width()
        label_height = self.label.winfo_height()

        # Resize the frame to fit the label's size
        test_img = cv2.resize(test_img, (label_width, label_height))

        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        if len(faces_detected) > 0:
            x, y, w, h = faces_detected[0]  # Get the first face's coordinates

            # Draw rectangle around the face
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

            roi_gray = gray_img[y:y + h, x:x + w]  # Crop the region of interest
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = self.model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            emoticon_map = {
                'angry': '😠', 'disgust': '🤢', 'fear': '😨', 
                'happy': '😄', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
            }
            emotion_text = predicted_emotion.capitalize() + ' ' + emoticon_map.get(predicted_emotion, '')

            # Convert OpenCV image to PIL for drawing text
            pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((x, y - 50), emotion_text, font=self.font, fill=(255, 0, 0, 255))

            # Convert back to OpenCV format
            test_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert image to Tkinter-compatible format
        img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        # Call show_frame again after 10 milliseconds to create a loop
        self.after(10, self.show_frame)

    def switch_to_second_page(self):
        """ Release the video capture and switch to the second page """
        self.cap.release()
        self.switch_page_callback()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
