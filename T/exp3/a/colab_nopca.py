# ============================================================
# Crop Leaf Disease Detection — Google Colab Training Script
# ============================================================
# Dataset: PlantVillage for Object Detection (YOLO format)
# https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo
#
# Methodology: 13 Features (RGB+Grayscale) + HOG (16x16) + Gaussian Blur → Random Forest
# NO PCA — raw features fed directly to Random Forest for axis-aligned splits
#
# HOW TO USE:
# 1. Open colab.research.google.com → New Notebook
# 2. Cell 1: !pip install -q kaggle scikit-learn opencv-python-headless matplotlib joblib pyyaml scikit-image "numpy<2" "pandas<2.0.0"
# 3. Cell 2: Paste this ENTIRE script → Run (Shift+Enter)
# 4. It will ask you to upload kaggle.json (one-time)
# 5. Wait → downloads results to your laptop automatically
# ============================================================

# ---- Upload Kaggle API token ----
from google.colab import files
print("Upload your kaggle.json file")
print("(Get it from: kaggle.com → My Account → API → Create New Token)")
uploaded = files.upload()

import os
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
os.rename("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# ---- Download the YOLO-format PlantVillage dataset ----
os.system("kaggle datasets download sebastianpalaciob/plantvillage-for-object-detection-yolo -p ./data --unzip")
print("\nDataset downloaded!")

# ---- Explore dataset structure ----
print("\nDataset structure:")
for root, dirs, fls in os.walk("./data"):
    depth = root.replace("./data", "").count(os.sep)
    if depth < 3:
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/ ({len(fls)} files)")
    if depth >= 3:
        break

# ---- Find the data.yaml to get class names ----
import glob, yaml

yaml_files = glob.glob("./data/**/*.yaml", recursive=True)
print(f"\nFound YAML configs: {yaml_files}")

class_names = []
if yaml_files:
    with open(yaml_files[0], "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get("names", [])
    print(f"Classes from YAML: {class_names}")
    print(f"Number of classes: {data_config.get('nc', len(class_names))}")

# If no YAML found, use default PlantVillage classes
if not class_names:
    class_names = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    print(f"Using default classes: {class_names}")

# ---- Find image and label folders ----
import cv2
import numpy as np

def find_folders(base, name):
    """Find all folders matching a name pattern."""
    results = []
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            if d == name:
                results.append(os.path.join(root, d))
    return results

# Find image folders
image_folders = find_folders("./data", "images")
if not image_folders:
    image_folders = glob.glob("./data/**/train", recursive=True)

print(f"\nImage folders found: {image_folders}")

# Find label folders
label_folders = find_folders("./data", "labels")
print(f"Label folders found: {label_folders}")


# ---- Feature Extraction (RGB + Grayscale + Blur + HOG 16x16, NO PCA) ----
from skimage.feature import hog

def extract_features(image):
    """Extract RGB, Grayscale Statistics, and HOG Textures with Gaussian Blur."""
    # 1. Blur the image to remove background noise (dirt, sky)
    img_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Resize to keep computations fast
    img_resized = cv2.resize(img_blurred, (128, 128))

    # 2. RGB Color Features
    b_mean, g_mean, r_mean = np.mean(img_resized, axis=(0, 1))
    b_std, g_std, r_std = np.std(img_resized, axis=(0, 1))

    # 3. Grayscale Statistical Features
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hist = np.histogram(gray, bins=256)[0]
    hist = hist / np.sum(hist)

    gray_features = [
        r_mean, g_mean, b_mean,
        r_std, g_std, b_std,
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.median(gray),
        np.var(gray),
        -np.sum(hist * np.log2(hist + 1e-7))
    ]

    # 4. HOG Textures at (16,16) — ~1,764 features, no PCA needed
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # Combine all features into one giant array
    return np.hstack([gray_features, hog_features])


def get_class_from_label(label_path, class_names):
    """Read a YOLO label file and return the most common class name."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None

    class_ids = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            try:
                class_ids.append(int(parts[0]))
            except ValueError:
                continue

    if not class_ids:
        return None

    most_common = max(set(class_ids), key=class_ids.count)
    if most_common < len(class_names):
        return class_names[most_common]
    return f"class_{most_common}"


# ---- Load all images and extract features ----
features = []
labels = []
class_counts = {}

# Process each image folder (train + val)
for img_folder in image_folders:
    # Find corresponding label folder
    label_folder = img_folder.replace("images", "labels")
    if not os.path.exists(label_folder):
        print(f"  No label folder for {img_folder}, skipping")
        continue

    print(f"\nProcessing: {img_folder}")
    for img_file in sorted(os.listdir(img_folder)):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_folder, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_folder, label_file)

        # Get class from YOLO label
        cls = get_class_from_label(label_path, class_names)
        if cls is None:
            continue

        # Read image and extract features
        image = cv2.imread(img_path)
        if image is None:
            continue

        feat = extract_features(image)
        features.append(feat)
        labels.append(cls)
        class_counts[cls] = class_counts.get(cls, 0) + 1

print(f"\n{'='*50}")
print(f"  DATASET SUMMARY")
print(f"{'='*50}")
print(f"Total samples: {len(features)}")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls}: {count} images")

# ---- Train/Test Split (NO PCA, NO Scaler — raw features) ----
from sklearn.model_selection import train_test_split

X, y = np.array(features), np.array(labels)
print(f"\nFeatures per sample: {X.shape[1]} (raw, no PCA)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---- Train Random Forest ----
from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest (500 trees, max_depth=25, balanced)...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---- Results ----
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"  ACCURACY: {accuracy:.2%}")
print(f"{'='*50}")

report = classification_report(y_test, y_pred)
print("\nClassification Report:\n")
print(report)

# Save report
with open("classification_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2%}\n\n")
    f.write(f"Dataset: PlantVillage for Object Detection (YOLO format)\n")
    f.write(f"Total samples: {len(X)}\n")
    f.write(f"Features per sample: {X.shape[1]} (raw HOG 16x16, no PCA)\n")
    f.write(f"Train: {len(X_train)}, Test: {len(X_test)}\n\n")
    f.write(report)

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(15, 12))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix — Random Forest (RGB + HOG 16x16, No PCA)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("  Saved confusion_matrix.png")

# ---- Save pipeline (model only, no scaler/PCA needed) ----
import joblib

model.n_jobs = 1

pipeline = {
    "model": model,
    "classes": model.classes_
}

joblib.dump(pipeline, "leaf_disease_pipeline.pkl", compress=3)
size_mb = os.path.getsize("leaf_disease_pipeline.pkl") / (1024 * 1024)
print(f"  Saved leaf_disease_pipeline.pkl ({size_mb:.1f} MB)")

# ---- Download everything ----
print("\nDownloading files to your laptop...")
files.download("confusion_matrix.png")
files.download("classification_report.txt")
files.download("leaf_disease_pipeline.pkl")

print("\nDONE! Output files downloaded:")
print("  confusion_matrix.png")
print("  classification_report.txt")
print("  leaf_disease_pipeline.pkl")
