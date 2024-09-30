import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk, ImageDraw, ImageFont  # Pillow modules for handling images
import os
import sys
import tempfile  # For creating temporary files


class PlaceholderPage(tk.Frame):
    """ A Tkinter Frame for uploading images and displaying detected facial expressions. """

    def __init__(self, parent, switch_page_callback):
        """ Initialize the PlaceholderPage with buttons and image display. """
        super().__init__(parent)
        self.switch_page_callback = switch_page_callback

        # Create a frame for buttons
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

        # Add buttons for switching to real-time detection and uploading images
        tk.Button(button_frame, text="Switch to Real Time Detection", command=self.switch_to_first_page).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)

        # Create a canvas for displaying the image
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Load the model and resources
        self.model = self.load_model()
        self.font = self.load_font()
        self.face_haar_cascade = self.load_cascade()

        self.image_on_canvas = None  # To hold the image on the canvas
        self.processed_image = None  # To hold the processed image with annotations
        self.canvas_scale = 1.0  # For zooming functionality

        # Bind mouse scroll event for zooming
        self.canvas.bind("<MouseWheel>", self.zoom)

    def resource_path(self, relative_path):
        """ Get the absolute path to the resource, works for development and PyInstaller. """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def load_model(self):
        """ Load the facial expression recognition model from JSON and weights. """
        model_json_path = self.resource_path("Facial Expression Recognition.json")
        model_weights_path = self.resource_path("fer.h5")

        with open(model_json_path, "r") as json_file:
            model = model_from_json(json_file.read())
        model.load_weights(model_weights_path)
        return model

    def load_cascade(self):
        """ Load Haar Cascade for face detection. """
        face_haar_cascade_path = self.resource_path('haarcascade_frontalface_default.xml')
        return cv2.CascadeClassifier(face_haar_cascade_path)

    def load_font(self):
        """ Load the font for emoji and text rendering. """
        font_path = self.resource_path("DejaVuSans.ttf")
        return ImageFont.truetype(font_path, 32)

    def upload_image(self):
        """ Prompt the user to upload an image and process it for facial expressions. """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            # Clear the canvas and reset any previous image-related variables
            self.canvas.delete("all")  # Clear the canvas
            self.processed_image = None  # Reset the processed image
            self.image_on_canvas = None  # Reset the image on canvas
            self.canvas_scale = 1.0  # Reset zoom scale
            self.process_image(file_path)  # Process the new image

    def process_image(self, file_path):
        """ Process the uploaded image for facial expression detection and display results. """
        test_img = cv2.imread(file_path)
        if test_img is None:
            messagebox.showerror("Error", "Could not read the image.")
            return

        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        if len(faces_detected) == 0:
            messagebox.showinfo("No Faces", "No faces were detected in the image.")
            return

        # Variable to store detected faces and their emotions
        rois = []

        for (x, y, w, h) in faces_detected:
            # Draw rectangle around detected face
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)

            # Prepare the face region for emotion prediction
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0) / 255.0  # Normalize the pixel values

            # Predict emotion
            predictions = self.model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            # Map emotion to corresponding emoticon
            emoticon_map = {
                'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                'happy': '😄', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
            }
            emotion_text = f"{predicted_emotion.capitalize()} {emoticon_map.get(predicted_emotion, '')}"

            # Store the face region and emotion text
            rois.append((x, y, w, h, emotion_text))

        # Display all detected faces with emotions
        self.display_image_with_emotions(test_img, rois)

    def display_image_with_emotions(self, test_img, rois):
        """ Display the processed image with detected emotion on the faces. """
        # Convert OpenCV image to PIL for drawing text
        pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for roi in rois:
            x, y, w, h, emotion_text = roi
            draw.text((x, y - 50), emotion_text, font=self.font, fill=(255, 0, 0, 255))

        # Save the processed image with the emotion text
        self.processed_image = pil_img

        # Update the canvas to display the processed image
        self.update_image_on_canvas(self.processed_image)

    def update_image_on_canvas(self, pil_img):
        """ Update the canvas with the new image and handle scaling. """
        self.canvas.delete("all")  # Clear previous images

        # Resize the image based on the current scale
        width, height = int(pil_img.width * self.canvas_scale), int(pil_img.height * self.canvas_scale)
        
        # Resize using LANCZOS for better quality
        pil_img = pil_img.resize((width, height), Image.LANCZOS)  

        # Convert the PIL image to a Tkinter-compatible image
        self.image_on_canvas = ImageTk.PhotoImage(pil_img)

        # Add the new image to the canvas
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor=tk.CENTER, image=self.image_on_canvas)

    def save_image(self):
        """ Save the processed image with emotions to a file. """
        if self.processed_image:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                self.processed_image.save(save_path)
                messagebox.showinfo("Image Saved", "Processed image saved successfully!")
        else:
            messagebox.showwarning("No Image", "No processed image to save.")

    def zoom(self, event):
        """ Zoom in and out of the processed image on mouse scroll. """
        if event.delta > 0:  # Scroll up
            self.canvas_scale *= 1.1  # Zoom in
        else:  # Scroll down
            self.canvas_scale /= 1.1  # Zoom out

        # Update the image on the canvas with the new scale
        if self.processed_image:
            self.update_image_on_canvas(self.processed_image)

    def switch_to_first_page(self):
        """ Switch to the first page of the application. """
        self.switch_page_callback()
