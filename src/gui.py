import cv2
import tkinter as tk
from tkinter import Tk, filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
from preprocessing import preprocessing
from feature_extraction import feature_extractor  # updated import

class AnimalDetectorGUI:
    def __init__(self, master, model, label_encoder):
        self.master = master
        self.model = model
        self.label_encoder = label_encoder
        

        self.master.title("Animal Detection System")

        self.label = Label(master, text="Animal Detection System", font=("Arial", 16))
        self.label.pack(pady=10)

        self.canvas = Canvas(master, width=300, height=300)
        self.canvas.pack()

        self.button = Button(master, text="Choose Image", command=self.choose_image)
        self.button.pack(pady=10)

        self.result_label = Label(master, text="", font=("Arial", 14), fg="blue")
        self.result_label.pack(pady=10)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Display image
            image = cv2.imread(file_path)
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(display_image)
            img_tk = ImageTk.PhotoImage(img_pil.resize((300, 300)))
            self.canvas.create_image(0, 0, anchor='nw', image=img_tk)
            self.canvas.image = img_tk

            # Preprocess and extract features
            processed = preprocessing(image)
            features = feature_extractor(processed).reshape(1, -1)

            # Predict class
            pred = self.model.predict(features)
            class_name = self.label_encoder.inverse_transform(pred)[0]

            # Display prediction
            self.result_label.config(text=f"Prediction: {class_name}")