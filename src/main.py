  # Entry point (runs GUI)
import tkinter as tk
from gui import AnimalDetectorGUI
from detector import load_trained_model

if __name__ == "__main__":
    # Load trained model, label encoder, and scaler
    model, label_encoder = load_trained_model()  # No need for scaler if not being used in the GUI
    root = tk.Tk()  # Create the Tkinter root window
    root.title("Object Detection System")  # Set the window title
    # Create the GUI and pass in the model and label encoder
    gui = AnimalDetectorGUI(root, model, label_encoder)
    root.mainloop()