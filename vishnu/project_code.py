!pip install opencv-python scikit-learn pandas numpy matplotlib

# -----------
import json

kaggle_creds = {
    "username": "vishnudharavath",
    "key": "yKGAT_64a92dbf78bb1fed91bfc1bf13744f85"
}

with open("kaggle.json", "w") as f:
    json.dump(kaggle_creds, f)

print("✅ kaggle.json created")

# -----------
!ls

# -----------
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# -----------
!kaggle datasets list | head

# -----------
!kaggle datasets download -d emmarex/plantdisease

# -----------
!unzip plantdisease.zip

# -----------
!ls

# -----------
!ls PlantVillage | head

# -----------
import os
import shutil

base = "PlantVillage"
target = "dataset"

classes = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight"
]

os.makedirs(target, exist_ok=True)

for cls in classes:
    src = os.path.join(base, cls)
    dst = os.path.join(target, cls)

    if not os.path.exists(src):
        print(f"❌ Missing folder: {cls}")
    elif not os.path.exists(dst):
        shutil.copytree(src, dst)
        print(f"✅ Copied: {cls}")
    else:
        print(f"⚠ Already exists: {cls}")

print("\n🎉 Dataset subset creation completed")

# -----------
for cls in os.listdir("dataset"):
    count = len(os.listdir(os.path.join("dataset", cls)))
    print(cls, "->", count, "images")

# -----------
!ls dataset

# -----------
import cv2
import numpy as np
import os
import pandas as pd

# -----------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# -----------
def extract_features(img):
    # Color features
    mean_rgb = np.mean(img, axis=(0, 1))
    std_rgb = np.std(img, axis=(0, 1))

    # Texture feature
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = np.std(gray)

    return np.hstack([mean_rgb, std_rgb, texture])

# -----------
X = []
y = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = preprocess_image(img_path)
        features = extract_features(img)

        X.append(features)
        y.append(label)

df = pd.DataFrame(X)
df["label"] = y

df.to_csv("leaf_features.csv", index=False)

print("✅ leaf_features.csv created successfully")
print("Dataset shape:", df.shape)

# -----------
df.head()

# -----------
df = pd.read_csv("leaf_features.csv")

# -----------
import pandas as pd
df = pd.read_csv("leaf_features.csv")

# -----------
X = df.drop("label", axis=1)
y = df["label"]

# -----------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Reduced features:", X_pca.shape[1])

# -----------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# -----------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------
!pip install scikit-image

# -----------
from skimage.feature import hog
import cv2
import numpy as np

def extract_hog_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

# -----------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------
import joblib
joblib.dump(rf, "leaf_disease_model.pkl")
print("✅ Model saved")

# -----------
import gradio as gr
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load trained model
model = joblib.load("leaf_disease_model.pkl")

def predict_leaf_disease(image):
    try:
        # Ensure image is valid
        if image is None:
            return "No image provided"

        # Resize
        img = cv2.resize(image, (128, 128))

        # Convert RGB (Gradio) → Grayscale safely
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Extract HOG features
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features = features.reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        return prediction

    except Exception as e:
        return f"Prediction error: {str(e)}"

# Gradio UI
interface = gr.Interface(
    fn=predict_leaf_disease,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=gr.Textbox(label="Predicted Disease"),
    title="🌿 Crop Leaf Disease Detection System",
    description="Upload a plant leaf image to identify the disease using a trained ML model."
)

interface.launch()

# -----------
!ls

# -----------
%%writefile requirements.txt
gradio
opencv-python-headless
scikit-learn
scikit-image
numpy
joblib

# -----------
!ls

# -----------
from google.colab import files
files.download("app.py")

# -----------
files.download("leaf_disease_model.pkl")

# -----------
files.download("requirements.txt")

# -----------
