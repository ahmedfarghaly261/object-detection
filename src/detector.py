# src/detector.py
from tensorflow.keras.models import load_model
from keras.models import load_model
import joblib
import numpy as np
import cv2
from preprocessing import preprocessing
from segmentation import extract_bounded, extract_rois
from feature_extraction import feature_extractor

def load_trained_model():
    model = load_model('models/animal_detection.h5')
    encoder = joblib.load('models/label_encoder(1).joblib')
    return model, encoder

# # Load model and encoder
# model = load_model('models/animal_detection.h5')
# encoder = joblib.load('models/label_encoder(1).joblib')

def predict_and_visualize(image, model, encoder, min_area=1200, margin=10):
    output_img = image.copy()
    processed_img = preprocessing(image)
    
    boxes = extract_bounded(processed_img, min_area, margin)
    if not boxes:
        print("Cannot find ROI")
        return []

    rois = extract_rois(image, boxes)
    predictions = []

    for (x1, y1, x2, y2), roi in zip(boxes, rois):
        processed_roi = preprocessing(roi)
        features = feature_extractor(processed_roi).reshape(1, -1)
        
        prediction = model.predict(features)
        predicted_class = encoder.inverse_transform(prediction)[0]
        predictions.append(predicted_class)

        # Draw box and label
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, predicted_class, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output_img, predictions
