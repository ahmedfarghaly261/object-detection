
# 🐾 Object Detection System for Animal Classification

This project is a complete object detection system that detects and classifies multiple animals (e.g., cats, dogs, etc.) within an image using a custom machine learning pipeline. The system includes preprocessing, segmentation, feature extraction (HOG), classification, and a GUI interface built with Tkinter.

## 📁 Project Structure

object-detection-system/
├── models/ # Contains trained model and label encoder
├── src/
│ ├── main.py # Entry point with GUI
│ ├── detector.py # Prediction and visualization logic
│ ├── preprocessing.py # Image preprocessing & augmentation
│ ├── segmentation.py # ROI segmentation and bounding box logic
│ ├── feature_extraction.py # HOG-based feature extractor
│ └── gui.py # Tkinter GUI for user interaction
├── images/ # Sample input images (optional)
├── README.md # Project documentation
├── .gitignore # Git ignore rules (e.g., ignoring venv/)
└── requirements.txt # List of required packages

bash
Copy
Edit


## 🧠 Features

- Detect multiple animals in a single image
- Preprocess images (resize, grayscale, enhance, normalize)
- Extract ROIs using segmentation
- Classify animals using a trained HOG + MLP model
- Simple and interactive GUI using Tkinter
- Augmentation support (rotation, flipping, etc.)

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/ahmedfarghaly261/object-detection.git
cd object-detection

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # For Windows

# Install dependencies
pip install -r requirements.txt

🚀 Running the App
python src/main.py

🧪 Dependencies
Python 3.x

OpenCV

NumPy

scikit-image

scikit-learn

TensorFlow / Keras

Tkinter (comes with Python)

📸 Example
Upload an image with animals, and the system will detect and label each animal in the image with a bounding box.

🧾 License
This project is for academic and educational use.



