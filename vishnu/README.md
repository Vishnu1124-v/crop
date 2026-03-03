# Vishnu — Crop Leaf Disease Detection: Complete Technical Documentation

## Overview

End-to-end crop leaf disease detection system: data download, feature extraction, model training, and live deployment as a Gradio web application on Hugging Face Spaces.

**Live App:** https://huggingface.co/spaces/chinni-1221/crop-leaf-disease-detection

---

## Dataset

- **Source:** PlantVillage dataset from Kaggle (`emmarex/plantdisease`)
- **Kaggle URL:** https://www.kaggle.com/datasets/emmarex/plantdisease
- **Download size:** 658 MB compressed (`plantdisease.zip`)
- **Format:** Standard folder format — one folder per class, images inside each folder
- **Original dataset:** Contains all PlantVillage classes (Pepper, Potato, Tomato, Apple, Corn, Grape, etc.)
- **Subset used:** 5 classes manually copied from the full dataset into a `dataset/` folder

### Classes Used (5-class subset)

| Class | Images | Notes |
|-------|--------|-------|
| Tomato_Late_blight | 1,909 | Largest class |
| Tomato_healthy | 1,591 | |
| Tomato_Early_blight | 1,000 | |
| Potato___Early_blight | 1,000 | |
| Potato___healthy | 152 | Smallest class (class imbalance) |
| **Total** | **5,652** | |

The subset was created by copying these 5 folders from the full `PlantVillage/` directory into a new `dataset/` directory using `shutil.copytree()`.

---

## Platform

- **Training environment:** Google Colab (free tier)
- **Runtime:** CPU (no GPU used)
- **Python version:** 3.12 (as shown in Colab pip output)
- **Libraries installed:** opencv-python, scikit-learn, pandas, numpy, matplotlib, scikit-image

---

## Preprocessing

Applied to every image before feature extraction:

1. **Read image:** `cv2.imread(img_path)` — loads as BGR color (3-channel, 128x128x3 after resize)
2. **Resize:** `cv2.resize(img, (128, 128))` — standardize all images to 128x128 pixels
3. **Gaussian Blur:** `cv2.GaussianBlur(img, (5, 5), 0)` — smooth noise with 5x5 kernel, sigma=0

```python
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img
```

---

## Feature Extraction (Training)

7 color-based features extracted from each preprocessed **BGR color** image:

```python
def extract_features(img):
    # Color features
    mean_rgb = np.mean(img, axis=(0, 1))   # 3 values: B, G, R channel means
    std_rgb = np.std(img, axis=(0, 1))     # 3 values: B, G, R channel stds

    # Texture feature
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = np.std(gray)                  # 1 value: grayscale std

    return np.hstack([mean_rgb, std_rgb, texture])  # 7 features total
```

| # | Feature | Formula | What It Captures |
|---|---------|---------|------------------|
| 1 | Blue channel mean | `np.mean(img, axis=(0,1))[0]` | Average blue intensity across all pixels |
| 2 | Green channel mean | `np.mean(img, axis=(0,1))[1]` | Average green intensity — diseased leaves lose green |
| 3 | Red channel mean | `np.mean(img, axis=(0,1))[2]` | Average red intensity — brown spots increase red |
| 4 | Blue channel std | `np.std(img, axis=(0,1))[0]` | Variation in blue — spots create non-uniform blue |
| 5 | Green channel std | `np.std(img, axis=(0,1))[1]` | Variation in green — patchy chlorosis |
| 6 | Red channel std | `np.std(img, axis=(0,1))[2]` | Variation in red — lesion borders vs healthy tissue |
| 7 | Grayscale texture std | `np.std(cv2.cvtColor(img, COLOR_BGR2GRAY))` | Overall brightness variation — more spots = higher std |

**Output:** Features saved to `leaf_features.csv` — shape **(5,652 rows x 8 columns)** (7 feature columns + 1 label column).

### Sample Data (first 5 rows from notebook output)

| Col 0 (B mean) | Col 1 (G mean) | Col 2 (R mean) | Col 3 (B std) | Col 4 (G std) | Col 5 (R std) | Col 6 (gray std) | Label |
|------|------|------|------|------|------|------|------|
| 121.07 | 121.21 | 128.34 | 22.78 | 19.15 | 21.22 | 18.86 | Tomato_Late_blight |
| 105.74 | 119.82 | 116.42 | 52.38 | 35.90 | 51.09 | 41.89 | Tomato_Late_blight |
| 123.49 | 125.27 | 127.06 | 30.67 | 26.64 | 27.30 | 26.45 | Tomato_Late_blight |
| 96.06 | 101.34 | 98.87 | 44.81 | 35.18 | 41.19 | 37.43 | Tomato_Late_blight |
| 118.81 | 127.36 | 125.21 | 56.65 | 43.89 | 46.57 | 45.31 | Tomato_Late_blight |

---

## Model 1: Gradient Boosting Classifier (with PCA)

### Pipeline

```
Raw 7 features → StandardScaler (zero mean, unit variance) → PCA (n_components=0.95) → GradientBoostingClassifier
```

### PCA Compression

- **Input:** 7 scaled features
- **n_components:** 0.95 (retain 95% of total variance)
- **Output:** **2 principal components** — PCA reduced 7 features to just 2

### Model Configuration

- **Algorithm:** GradientBoostingClassifier (scikit-learn default hyperparameters)
- **random_state:** 42
- **All other parameters:** scikit-learn defaults (n_estimators=100, learning_rate=0.1, max_depth=3)

### Train/Test Split

- **Split ratio:** 80% train / 20% test
- **Stratified:** Yes (`stratify=y`)
- **random_state:** 42
- **Test set size:** 1,131 samples

### Results

**Accuracy: 62.86%** (0.6286472148541115)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Potato___Early_blight | 0.65 | 0.82 | 0.73 | 200 |
| Potato___healthy | 0.00 | 0.00 | 0.00 | 31 |
| Tomato_Early_blight | 0.50 | 0.45 | 0.47 | 200 |
| Tomato_Late_blight | 0.63 | 0.52 | 0.57 | 382 |
| Tomato_healthy | 0.69 | 0.81 | 0.74 | 318 |
| | | | | |
| **Accuracy** | | | **0.63** | **1,131** |
| **Macro avg** | **0.49** | **0.52** | **0.50** | **1,131** |
| **Weighted avg** | **0.61** | **0.63** | **0.61** | **1,131** |

**Key observations:**
- Potato___healthy collapsed to 0% precision, 0% recall, 0% F1 — the model never predicted this class
- Only 31 test samples for Potato___healthy (smallest class) + PCA compressed away distinguishing information
- PCA reduced 7 features to just 2 — too aggressive for a tree-based model
- Best class: Potato___Early_blight (82% recall) and Tomato_healthy (81% recall)

---

## Model 2: Random Forest Classifier (Final Deployed Model)

### Pipeline

```
Raw 7 features → RandomForestClassifier (no PCA, no scaling)
```

### Model Configuration

- **Algorithm:** RandomForestClassifier (scikit-learn)
- **n_estimators:** 500 (500 decision trees)
- **class_weight:** "balanced" (auto-adjusts weights inversely proportional to class frequency)
- **random_state:** 42
- **n_jobs:** -1 (uses all available CPU cores)
- **max_depth:** None (default — unlimited tree depth)
- **All other parameters:** scikit-learn defaults

### Train/Test Split

- **Split ratio:** 80% train / 20% test
- **Stratified:** Yes (`stratify=y`)
- **random_state:** 42
- **Training set:** 4,521 samples
- **Test set:** 1,131 samples

### Results

**Accuracy: 83.90%** (0.8390804597701149)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Potato___Early_blight | 0.87 | 0.91 | 0.89 | 200 |
| Potato___healthy | 0.85 | 0.71 | 0.77 | 31 |
| Tomato_Early_blight | 0.76 | 0.71 | 0.74 | 200 |
| Tomato_Late_blight | 0.80 | 0.79 | 0.79 | 382 |
| Tomato_healthy | 0.90 | 0.95 | 0.93 | 318 |
| | | | | |
| **Accuracy** | | | **0.84** | **1,131** |
| **Macro avg** | **0.84** | **0.81** | **0.82** | **1,131** |
| **Weighted avg** | **0.84** | **0.84** | **0.84** | **1,131** |

**Key observations:**
- 21-point accuracy gain over Gradient Boosting (83.9% vs 62.9%)
- Potato___healthy recovered from 0% to 71% recall — Random Forest on raw features handles small classes better than PCA+GradientBoosting
- Tomato_healthy is the best class (95% recall) — healthy leaves are visually distinct
- Tomato_Early_blight is the weakest class (71% recall) — early blight has subtle visual symptoms
- class_weight="balanced" helped compensate for the Potato___healthy imbalance (152 vs 1,909 images)

### Model Saved

```python
import joblib
joblib.dump(rf, "leaf_disease_model.pkl")
```

Saved as `leaf_disease_model.pkl` using joblib default compression.

---

## Unused HOG Code

The notebook defines a HOG feature extraction function in cell 24 that is **never called during training**:

```python
from skimage.feature import hog

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
```

This function exists in `project_code.py` but the training in cell 25 uses the original `df` (7 color features from `leaf_features.csv`), not HOG features. The model was trained on 7 color features only.

---

## Deployment

### Hugging Face Spaces Configuration

```yaml
---
title: Crop Leaf Disease Detection
emoji: eye
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---
```

**Space URL:** https://huggingface.co/spaces/chinni-1221/crop-leaf-disease-detection

### app.py — Deployed Inference Code

The deployed `app.py` uses a **completely different feature extraction** than the training code:

```python
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Note: RGB input (Gradio), not BGR
    resized = cv2.resize(gray, (128, 128))

    mean = np.mean(resized)
    std = np.std(resized)
    min_val = np.min(resized)
    max_val = np.max(resized)
    median = np.median(resized)
    variance = np.var(resized)

    hist = np.histogram(resized, bins=256)[0]
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    return [mean, std, min_val, max_val, median, variance, entropy]
```

| # | Feature | Code |
|---|---------|------|
| 1 | Mean intensity | `np.mean(resized)` |
| 2 | Standard deviation | `np.std(resized)` |
| 3 | Minimum pixel value | `np.min(resized)` |
| 4 | Maximum pixel value | `np.max(resized)` |
| 5 | Median pixel value | `np.median(resized)` |
| 6 | Variance | `np.var(resized)` |
| 7 | Entropy | `-np.sum(hist * np.log2(hist + 1e-7))` from 256-bin histogram |

### Known Issue: Training vs Inference Feature Mismatch

| | Training (`project_code.py`) | Inference (`app.py`) |
|---|---|---|
| Input image | BGR color (OpenCV default) | RGB (Gradio provides RGB) |
| Image used for features | **Color** (3-channel BGR) | **Grayscale** (single channel) |
| Gaussian blur | Yes (5,5) | No |
| Feature 1 | Blue channel mean | Grayscale mean |
| Feature 2 | Green channel mean | Grayscale std |
| Feature 3 | Red channel mean | Grayscale min |
| Feature 4 | Blue channel std | Grayscale max |
| Feature 5 | Green channel std | Grayscale median |
| Feature 6 | Red channel std | Grayscale variance |
| Feature 7 | Grayscale texture std | Grayscale entropy |

The model was trained on 7 color features but receives 7 different grayscale features at inference time. These are entirely different feature spaces — the numbers in each position have different meanings and ranges.

Additionally, the notebook cell 27 writes a **third** version of app.py that uses HOG features for prediction — but this is not the same as the `app.py` file that was actually deployed to HuggingFace.

### Gradio Interface

```python
interface = gr.Interface(
    fn=predict_leaf_disease,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=gr.Textbox(label="Predicted Disease"),
    title="Crop Leaf Disease Detection System",
    description="Upload a leaf image to identify the disease using a trained ML model."
)
```

- **Input:** User uploads an image (received as numpy array in RGB format)
- **Output:** Text label — one of the 5 class names
- **Error handling:** Try/except wraps the prediction, returns error message string on failure

### Requirements

```
gradio
opencv-python-headless
scikit-learn
scikit-image
numpy
joblib
```

Note: `scikit-image` is listed but not used by the deployed `app.py` (it was in the notebook's HOG code that isn't used). `opencv-python-headless` is used instead of `opencv-python` because HuggingFace Spaces has no display server.

### HuggingFace Git LFS Configuration

The `gitattributes` file in this folder configures LFS for the HuggingFace Space repository:

```
*.pkl filter=lfs diff=lfs merge=lfs -text
*.7z filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
... (and 25+ other binary file extensions)
```

---

## Files

| File | Description |
|------|-------------|
| `project_code.py` | Full training pipeline — Colab notebook cells exported as .py with `# -----------` cell separators. Contains: Kaggle setup, dataset download, subset creation, feature extraction, Gradient Boosting training, Random Forest training, HOG function (unused), model export, Gradio app writing, requirements writing, file downloads. |
| `Project.ipynb` | Jupyter notebook with all cell outputs — same code as `project_code.py` but with interactive outputs showing: pip install logs, Kaggle download progress (658MB), dataset class counts, feature DataFrame head, both model classification reports, app.py/requirements.txt file writing confirmations. |
| `app.py` | Deployed Gradio web app — loads `leaf_disease_model.pkl`, extracts 7 grayscale features from uploaded image, returns predicted class name. This is the file running on HuggingFace Spaces. |
| `leaf_disease_model.pkl` | Trained Random Forest model — 500 trees, 5 classes, trained on 7 color features, serialized with joblib. Tracked by Git LFS. |
| `requirements (1).txt` | Python dependencies for HuggingFace Spaces deployment: gradio, opencv-python-headless, scikit-learn, scikit-image, numpy, joblib. |
| `gitattributes` | HuggingFace LFS configuration — tracks .pkl, .h5, .pt, .tflite, .bin, and 25+ other binary extensions via Git LFS. |
| `README.md` | This file. |

---

## Code Pipeline (Step-by-Step from Project.ipynb)

| Step | Cell | Action | Output |
|------|------|--------|--------|
| 1 | 0 | `!pip install opencv-python scikit-learn pandas numpy matplotlib` | All already satisfied (Colab pre-installs) |
| 2 | 1 | Create `kaggle.json` with API credentials | "kaggle.json created" |
| 3 | 3 | Copy kaggle.json to `~/.kaggle/`, set permissions 600 | (no output) |
| 4 | 4 | `!kaggle datasets list` | Shows top Kaggle datasets |
| 5 | 5 | `!kaggle datasets download -d emmarex/plantdisease` | Downloads 658 MB zip |
| 6 | 6 | `!unzip plantdisease.zip` | Extracts PlantVillage folder |
| 7 | 9 | Copy 5 class folders from PlantVillage/ to dataset/ | "Copied: Tomato_healthy" x5 |
| 8 | 10 | Count images per class | Tomato_Late_blight=1909, Potato___healthy=152, etc. |
| 9 | 12-14 | Define `preprocess_image()` and `extract_features()` | (functions defined) |
| 10 | 15 | Loop all images, extract features, save to CSV | "leaf_features.csv created", shape (5652, 8) |
| 11 | 17-18 | Reload CSV into DataFrame | (df loaded) |
| 12 | 19 | Split X (features) and y (labels) | 7 feature columns, 1 label column |
| 13 | 20 | StandardScaler + PCA (95% variance) | "Reduced features: 2" |
| 14 | 21 | Train/test split (80/20, stratified) | 4,521 train / 1,131 test |
| 15 | 22 | Train GradientBoostingClassifier, predict, report | Accuracy: 62.86% |
| 16 | 23 | `!pip install scikit-image` | Already satisfied |
| 17 | 24 | Define `extract_hog_features()` (HOG) | (function defined, never called) |
| 18 | 25 | Train RandomForestClassifier (500 trees, balanced) on raw 7 features | Accuracy: 83.90% |
| 19 | 26 | `joblib.dump(rf, "leaf_disease_model.pkl")` | "Model saved" |
| 20 | 27 | Define Gradio interface with HOG-based prediction | (writes app.py with HOG code) |
| 21 | 29 | `%%writefile requirements.txt` | Writes requirements file |
| 22 | 31-33 | `files.download()` for app.py, model, requirements | Downloads to local machine |

**Note on cell 27:** The notebook's cell 27 creates a Gradio app that uses HOG features for prediction. However, the actual `app.py` file deployed to HuggingFace uses grayscale statistical features instead. The deployed `app.py` was likely edited separately after download from Colab.

---

## Libraries Used

| Library | Version (Colab) | Purpose |
|---------|-----------------|---------|
| scikit-learn | 1.6.1 | RandomForestClassifier, GradientBoostingClassifier, PCA, StandardScaler, train_test_split, accuracy_score, classification_report |
| OpenCV (cv2) | 4.13.0.90 | Image reading (`imread`), resizing, Gaussian blur, BGR-to-grayscale conversion |
| NumPy | 2.0.2 | Array operations, `mean`, `std`, `hstack` for feature vectors |
| pandas | 2.2.2 | DataFrame for features, CSV read/write (`to_csv`, `read_csv`) |
| matplotlib | 3.10.0 | Installed but not explicitly used in the notebook cells |
| scikit-image | 0.25.2 | HOG feature extraction (`skimage.feature.hog`) — imported but the function is never called during training |
| joblib | 1.5.3 | Model serialization (`joblib.dump`, `joblib.load`) |
| Gradio | 6.5.1 | Web UI for inference on HuggingFace Spaces |
