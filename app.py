import gradio as gr
import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("leaf_disease_model.pkl")

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (128, 128))

    # Feature 1: Mean intensity
    mean = np.mean(resized)

    # Feature 2: Std deviation
    std = np.std(resized)

    # Feature 3: Min pixel
    min_val = np.min(resized)

    # Feature 4: Max pixel
    max_val = np.max(resized)

    # Feature 5: Median
    median = np.median(resized)

    # Feature 6: Variance
    variance = np.var(resized)

    # Feature 7: Entropy (simple approximation)
    hist = np.histogram(resized, bins=256)[0]
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    return [mean, std, min_val, max_val, median, variance, entropy]

def predict_leaf_disease(image):
    try:
        features = extract_features(image)
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        return f"Prediction error: {str(e)}"

interface = gr.Interface(
    fn=predict_leaf_disease,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=gr.Textbox(label="Predicted Disease"),
    title="🌿 Crop Leaf Disease Detection System",
    description="Upload a leaf image to identify the disease using a trained ML model."
)

interface.launch()
