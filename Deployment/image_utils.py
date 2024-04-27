# image_utils.py

from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import cv2


# Function to load the trained Random Forest model
def load_model(model_path):
    return joblib.load(model_path)

def load_decoder(decoder_path):
    return joblib.load(decoder_path)

def extract_hog_features(image_path):
    try:
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(gray_image)
        if hog_features is None:
            print(f"Error computing HOG features for image: {image_path}")
            return None
        return hog_features.flatten()
    except Exception as e:
        print(f"Error processing image: {image_path} - {e}")
        return None



# Function to preprocess an image for model input
def preprocess_image(test_image_path):
    test_image = extract_hog_features(test_image_path)
    test_image = test_image.reshape(-1, 1)
    y_test = 1
    test_image = test_image[:y_test]
    img_array = test_image.reshape(1, -1)
    return img_array

# Function to make a prediction using the loaded model
def predict_image(model,label_encoder, img_array):
    prediction = model.predict(img_array)
    decoded_labels = label_encoder.inverse_transform(prediction)
    return decoded_labels

