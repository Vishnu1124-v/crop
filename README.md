# Crop Leaf Disease Detection — Complete Project Documentation

## Project Overview

Crop leaf disease detection using the PlantVillage dataset with traditional machine learning (Random Forest) and deep learning (MobileNetV3-Small). Two team members worked on complementary aspects:

- **Vishnu:** Built a working 5-class web application deployed on Hugging Face Spaces
- **T:** Conducted progressive experiments on all 38 classes for an Edge AI research paper

**Live Demo:** https://huggingface.co/spaces/chinni-1221/crop-leaf-disease-detection

---

## Project Results Summary

| Experiment / Approach | Team Member | Classes | Accuracy | Model Size | Inference Latency | Hardware | Notes |
|-----------------------|-------------|---------|----------|------------|-------------------|----------|-------|
| **Gradient Boosting + PCA** | Vishnu | 5 | 62.86% | < 5 MB | — | CPU | PCA compressed 7 features to 2, causing 0% recall on smallest class |
| **Random Forest (7 features)** | Vishnu | 5 | 83.90% | ~50 MB | — | CPU | Best 5-class model. Deployed to Hugging Face Spaces |
| **Experiment 1: 7-feature Baseline** | T | 38 | 44.48% | 699.8 MB | — | CPU | 7 grayscale stats are insufficient for 38 classes |
| **Experiment 2: HOG + PCA** | T | 38 | 53.55% | — | — | CPU | PCA rotates features diagonally, breaking Random Forest axis-aligned splits |
| **Experiment 3a: HOG (16x16) depth=25** | T | 38 | 75.52% | 445.6 MB | 22.40 ms/image | CPU | Removing PCA yielded a 22-point accuracy jump |
| **Experiment 3b: HOG (16x16) depth=None**| T | 38 | 75.31% | 511.1 MB | 22.65 ms/image | CPU | Proved ~75% is the hard feature ceiling for hand-crafted features on 38 classes |
| **Experiment 4: MobileNetV3-Small** | T | 38 | 98.13% | 1.1 MB | 5.54 ms/image | CPU (TFLite) | End-to-end deep learning broke the 75% feature ceiling |

### Edge AI Feasibility Conclusion

For a 5-class subset, hand-crafted features and a Random Forest (**Vishnu's model**) works beautifully on CPU edge devices. However, scaling to 38 classes (**T's experiments**) reveals a hard ~75% feature ceiling for traditional ML. To achieve production accuracy on 38 classes for Edge AI devices, a lightweight deep learning model like **MobileNetV3-Small** is required, offering 98.13% accuracy in a 1.1 MB `.tflite` deployment package. A Hybrid CNN+RF approach was considered and rejected due to "two brains" software complexity and bloated model size.

**Measured Inference Latency (SageMaker ml.g4dn.xlarge, 200 images, CPU):**
- Exp 3a (HOG+RF, depth=25): **22.40 ms/image** (median: 22.35 ms)
- Exp 3b (HOG+RF, depth=None): **22.65 ms/image** (median: 22.58 ms)
- Exp 4 (MobileNetV3, TFLite CPU): **5.54 ms/image** (median: 5.54 ms) — 4x faster than traditional ML, in a 1.1 MB package

---

## Datasets

Two versions of the PlantVillage dataset are used:

| | Vishnu | T |
|---|---|---|
| **Kaggle source** | `emmarex/plantdisease` | `sebastianpalaciob/plantvillage-for-object-detection-yolo` |
| **Kaggle URL** | https://www.kaggle.com/datasets/emmarex/plantdisease | https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo |
| **Format** | Standard folder (images in class folders) | YOLO (each image has a `.txt` label file with `class_id x_center y_center width height`) |
| **Class names source** | Folder names | `classes.yaml` inside the dataset |
| **Classes** | 5 (Potato + Tomato subset) | 38 (all plant diseases) |
| **Total images** | 5,652 | 54,293 |
| **Download size** | 658 MB compressed | ~829 MB compressed |
| **License** | Unknown | CC-BY-NC-SA-4.0 |
| **Class imbalance** | Mild (152 to 1,909 per class) | Severe (36:1 ratio, 152 to 5,507) |

### Vishnu's 5 Classes

| Class | Images | Notes |
|-------|--------|-------|
| Tomato_Late_blight | 1,909 | Largest class |
| Tomato_healthy | 1,591 | |
| Tomato_Early_blight | 1,000 | |
| Potato___Early_blight | 1,000 | |
| Potato___healthy | 152 | Smallest class (12.6:1 imbalance with largest) |
| **Total** | **5,652** | |

The subset was created by copying these 5 folders from the full `PlantVillage/` directory into a new `dataset/` directory using `shutil.copytree()`.

### T's 38 Classes (Full Class Distribution)

| Class | Samples | | Class | Samples |
|-------|---------|---|-------|---------|
| Apple___Apple_scab | 630 | | Orange___Haunglongbing_(Citrus_greening) | 5,507 |
| Apple___Black_rot | 621 | | Peach___Bacterial_spot | 2,297 |
| Apple___Cedar_apple_rust | 275 | | Peach___healthy | 360 |
| Apple___healthy | 1,645 | | Pepper,_bell___Bacterial_spot | 997 |
| Blueberry___healthy | 1,502 | | Pepper,_bell___healthy | 1,477 |
| Cherry___Powdery_mildew | 1,052 | | Potato___Early_blight | 1,000 |
| Cherry___healthy | 853 | | Potato___Late_blight | 1,000 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 513 | | Potato___healthy | 152 |
| Corn___Common_rust | 1,192 | | Raspberry___healthy | 371 |
| Corn___Northern_Leaf_Blight | 983 | | Soybean___healthy | 5,089 |
| Corn___healthy | 1,157 | | Squash___Powdery_mildew | 1,835 |
| Grape___Black_rot | 1,180 | | Strawberry___Leaf_scorch | 1,109 |
| Grape___Esca_(Black_Measles) | 1,383 | | Strawberry___healthy | 456 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1,076 | | Tomato___Bacterial_spot | 2,127 |
| Grape___healthy | 423 | | Tomato___Early_blight | 1,000 |
| Tomato___Late_blight | 1,907 | | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,357 |
| Tomato___Leaf_Mold | 952 | | Tomato___Tomato_mosaic_virus | 373 |
| Tomato___Septoria_leaf_spot | 1,771 | | Tomato___healthy | 1,591 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 | | | |
| Tomato___Target_Spot | 1,404 | | | |

**Imbalance range:** 36:1 ratio (Potato___healthy: 152 vs Orange___Haunglongbing: 5,507)

---

## Vishnu's Work — Complete Details

**Folder:** `vishnu/`
**Platform:** Google Colab (free tier, CPU)
**Python:** 3.12
**Goal:** Build a working web app for 5-class Potato/Tomato disease detection

### Vishnu — Preprocessing

Applied to every image before feature extraction:

```python
def preprocess_image(img_path):
    img = cv2.imread(img_path)           # Load as BGR color
    img = cv2.resize(img, (128, 128))    # Standardize to 128x128
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Smooth noise
    return img
```

1. **Read:** `cv2.imread(img_path)` — loads as BGR color (3-channel)
2. **Resize:** 128x128 pixels
3. **Gaussian Blur:** (5,5) kernel, sigma=0

### Vishnu — Feature Extraction (Training)

7 color-based features from each preprocessed **BGR color** image:

```python
def extract_features(img):
    mean_rgb = np.mean(img, axis=(0, 1))   # 3 values: B, G, R channel means
    std_rgb = np.std(img, axis=(0, 1))     # 3 values: B, G, R channel stds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = np.std(gray)                  # 1 value: grayscale std
    return np.hstack([mean_rgb, std_rgb, texture])  # 7 total
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

Features saved to `leaf_features.csv` — shape **(5,652 x 8)**: 7 feature columns + 1 label column.

#### Sample Data (first 5 rows from notebook output)

| B mean | G mean | R mean | B std | G std | R std | Gray std | Label |
|--------|--------|--------|-------|-------|-------|----------|-------|
| 121.07 | 121.21 | 128.34 | 22.78 | 19.15 | 21.22 | 18.86 | Tomato_Late_blight |
| 105.74 | 119.82 | 116.42 | 52.38 | 35.90 | 51.09 | 41.89 | Tomato_Late_blight |
| 123.49 | 125.27 | 127.06 | 30.67 | 26.64 | 27.30 | 26.45 | Tomato_Late_blight |
| 96.06 | 101.34 | 98.87 | 44.81 | 35.18 | 41.19 | 37.43 | Tomato_Late_blight |
| 118.81 | 127.36 | 125.21 | 56.65 | 43.89 | 46.57 | 45.31 | Tomato_Late_blight |

### Vishnu — Model 1: Gradient Boosting + PCA

**Pipeline:** `Raw 7 features -> StandardScaler -> PCA (n_components=0.95) -> GradientBoostingClassifier`

**PCA:** Reduced 7 features to **2 principal components** (95% variance threshold)

**Model config:** GradientBoostingClassifier with scikit-learn defaults (n_estimators=100, learning_rate=0.1, max_depth=3), random_state=42

**Train/test:** 80/20 stratified, random_state=42, test set: 1,131 samples

**Accuracy: 62.86%** (0.6286472148541115)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Potato___Early_blight | 0.65 | 0.82 | 0.73 | 200 |
| Potato___healthy | 0.00 | 0.00 | 0.00 | 31 |
| Tomato_Early_blight | 0.50 | 0.45 | 0.47 | 200 |
| Tomato_Late_blight | 0.63 | 0.52 | 0.57 | 382 |
| Tomato_healthy | 0.69 | 0.81 | 0.74 | 318 |
| **Accuracy** | | | **0.63** | **1,131** |
| **Macro avg** | **0.49** | **0.52** | **0.50** | **1,131** |
| **Weighted avg** | **0.61** | **0.63** | **0.61** | **1,131** |

Potato___healthy collapsed to 0/0/0 — model never predicted this class. PCA compressed 7 features to just 2, losing class-separating information.

### Vishnu — Model 2: Random Forest (Final Deployed Model)

**Pipeline:** `Raw 7 features -> RandomForestClassifier (no PCA, no scaling)`

**Model config:**
- **Algorithm:** RandomForestClassifier (scikit-learn)
- **n_estimators:** 500
- **class_weight:** "balanced"
- **random_state:** 42
- **n_jobs:** -1 (all CPU cores)
- **max_depth:** None (unlimited)

**Train/test:** 80/20 stratified, random_state=42, training: 4,521, test: 1,131 samples

**Accuracy: 83.90%** (0.8390804597701149)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Potato___Early_blight | 0.87 | 0.91 | 0.89 | 200 |
| Potato___healthy | 0.85 | 0.71 | 0.77 | 31 |
| Tomato_Early_blight | 0.76 | 0.71 | 0.74 | 200 |
| Tomato_Late_blight | 0.80 | 0.79 | 0.79 | 382 |
| Tomato_healthy | 0.90 | 0.95 | 0.93 | 318 |
| **Accuracy** | | | **0.84** | **1,131** |
| **Macro avg** | **0.84** | **0.81** | **0.82** | **1,131** |
| **Weighted avg** | **0.84** | **0.84** | **0.84** | **1,131** |

Key: 21-point gain over Gradient Boosting. Potato___healthy recovered from 0% to 71% recall. Tomato_healthy best at 95% recall. Tomato_Early_blight weakest at 71% recall. Model saved as `leaf_disease_model.pkl` via `joblib.dump()`.

### Vishnu — Unused HOG Code

The notebook defines `extract_hog_features()` (HOG with orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys') that is **never called during training**. Cell 25 trains the RF on the original 7 color features from `leaf_features.csv`, not on HOG features.

### Vishnu — Deployment on Hugging Face Spaces

- **URL:** https://huggingface.co/spaces/chinni-1221/crop-leaf-disease-detection
- **SDK:** Gradio 6.5.1
- **App file:** `app.py`
- **Model file:** `leaf_disease_model.pkl`

#### Deployed app.py Feature Extraction (Inference)

The deployed `app.py` extracts 7 **grayscale** statistical features (different from training):

```python
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGB from Gradio
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

| # | app.py Feature (Inference) | Training Feature | Match? |
|---|---|---|---|
| 1 | Grayscale mean | Blue channel mean | No |
| 2 | Grayscale std | Green channel mean | No |
| 3 | Grayscale min pixel | Red channel mean | No |
| 4 | Grayscale max pixel | Blue channel std | No |
| 5 | Grayscale median | Green channel std | No |
| 6 | Grayscale variance | Red channel std | No |
| 7 | Grayscale entropy | Grayscale texture std | No |

**Known issue:** The model was trained on 7 color features but the deployed app.py sends 7 completely different grayscale features. No Gaussian blur in app.py either. The notebook cell 27 also writes a third version of app.py using HOG features — but that is not what's deployed.

#### Gradio Interface

```python
interface = gr.Interface(
    fn=predict_leaf_disease,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=gr.Textbox(label="Predicted Disease"),
    title="Crop Leaf Disease Detection System",
    description="Upload a leaf image to identify the disease using a trained ML model."
)
```

Input: numpy array (RGB). Output: class name string. Error handling via try/except.

#### Deployment Requirements

```
gradio
opencv-python-headless
scikit-learn
scikit-image
numpy
joblib
```

`opencv-python-headless` (no GUI) instead of `opencv-python`. `scikit-image` listed but not used by the deployed app.py.

### Vishnu — Code Pipeline (Step-by-Step from Project.ipynb)

| Step | Cell | Action | Output |
|------|------|--------|--------|
| 1 | 0 | `!pip install opencv-python scikit-learn pandas numpy matplotlib` | All already satisfied |
| 2 | 1 | Create `kaggle.json` with API credentials | "kaggle.json created" |
| 3 | 3 | Copy kaggle.json to `~/.kaggle/`, chmod 600 | (no output) |
| 4 | 4 | `!kaggle datasets list \| head` | Shows top datasets |
| 5 | 5 | `!kaggle datasets download -d emmarex/plantdisease` | Downloads 658 MB |
| 6 | 6 | `!unzip plantdisease.zip` | Extracts PlantVillage/ |
| 7 | 9 | Copy 5 class folders to dataset/ | "Copied: ..." x5 |
| 8 | 10 | Count images per class | 5 counts printed |
| 9 | 12-14 | Define preprocess_image() and extract_features() | (defined) |
| 10 | 15 | Extract features for all 5,652 images -> CSV | "leaf_features.csv created", shape (5652, 8) |
| 11 | 17-18 | Reload CSV | (loaded) |
| 12 | 19 | Split X/y | 7 features, 1 label |
| 13 | 20 | StandardScaler + PCA(0.95) | "Reduced features: 2" |
| 14 | 21 | Train/test split 80/20 stratified | 4,521 / 1,131 |
| 15 | 22 | GradientBoostingClassifier -> predict -> report | 62.86% |
| 16 | 23 | `!pip install scikit-image` | Already satisfied |
| 17 | 24 | Define extract_hog_features() | (defined, never called) |
| 18 | 25 | RF(500, balanced) on raw 7 features -> predict -> report | 83.90% |
| 19 | 26 | joblib.dump(rf, "leaf_disease_model.pkl") | "Model saved" |
| 20 | 27 | Define Gradio interface (HOG-based) | (writes HOG app.py) |
| 21 | 29 | %%writefile requirements.txt | Writes file |
| 22 | 31-33 | files.download() x3 | Downloads to local |

### Vishnu — Files

| File | Size | Description |
|------|------|-------------|
| `project_code.py` | — | Full training pipeline (Colab cells as .py with `# -----------` separators) |
| `Project.ipynb` | — | Jupyter notebook with all cell outputs |
| `app.py` | — | Deployed Gradio web app (7 grayscale features) |
| `leaf_disease_model.pkl` | — | Trained RF model (500 trees, 5 classes, Git LFS) |
| `requirements (1).txt` | — | Deployment dependencies |
| `gitattributes` | — | HuggingFace LFS config (25+ binary extensions) |
| `README.md` | — | Vishnu's detailed technical README |

---

## T's Work — Complete Details

**Folder:** `T/` (organized into `exp1/`, `exp2/`, `exp3/`, `exp4/`, `work/`)
**Goal:** Progressive experiments for Edge AI research paper on all 38 PlantVillage classes

### T — Common Configuration (All Experiments)

| Parameter | Value |
|-----------|-------|
| Image resize | 128x128 pixels |
| Histogram bins | 256 (normalized to probability distribution) |
| Classifier | Random Forest (scikit-learn) |
| n_estimators | 500 (500 decision trees) |
| class_weight | "balanced" (auto-adjusts weights inversely proportional to class frequency) |
| random_state | 42 |
| n_jobs | -1 (all available CPU cores) |
| Train/test split | 80% train / 20% test |
| Stratified | Yes (preserves class proportions) |
| Training set | 43,434 images |
| Test set | 10,859 images |
| Label reading | YOLO format — reads `class_id` from `.txt` label files, maps to class name via `classes.yaml` |
| Model serialization | joblib with compress=3 |

---

### T — Experiment 1: 7 Grayscale Features (Baseline)

**Folder:** `T/exp1/`
**Files:** `Crop.ipynb`, `crop_code.py`, `train_model.py`
**Platform:** Google Colab (free tier, CPU)

#### Preprocessing

1. Convert image from BGR to grayscale (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`)
2. Resize to **128x128** pixels
3. Compute 256-bin histogram, normalize to probability distribution

**No Gaussian blur in this experiment.**

#### Features Extracted (7 per image)

| # | Feature | Formula | What It Captures |
|---|---------|---------|------------------|
| 1 | Mean pixel intensity | `np.mean(gray)` | Average brightness |
| 2 | Standard deviation | `np.std(gray)` | Spread of pixel values — spots = high std |
| 3 | Minimum pixel value | `np.min(gray)` | Darkest pixel — necrotic/dead tissue |
| 4 | Maximum pixel value | `np.max(gray)` | Brightest pixel — chlorosis/yellowing |
| 5 | Median pixel value | `np.median(gray)` | Robust center value, less affected by outliers |
| 6 | Variance | `np.var(gray)` | Like std but amplified — separates healthy vs diseased |
| 7 | Entropy | `-sum(hist * log2(hist + 1e-7))` | Texture complexity — diseased = more complex |

**No color features. No texture features. Only grayscale statistics.**

#### Model Configuration

- **max_depth:** None (unlimited tree growth)
- Everything else: common configuration above

#### Results — Accuracy: 44.48%

**Full 38-class classification report:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Apple___Apple_scab | 0.29 | 0.07 | 0.11 | 126 |
| Apple___Black_rot | 0.21 | 0.18 | 0.19 | 124 |
| Apple___Cedar_apple_rust | 0.20 | 0.04 | 0.06 | 55 |
| Apple___healthy | 0.36 | 0.21 | 0.27 | 329 |
| Blueberry___healthy | 0.53 | 0.61 | 0.57 | 300 |
| Cherry___Powdery_mildew | 0.23 | 0.09 | 0.12 | 210 |
| Cherry___healthy | 0.65 | 0.55 | 0.60 | 171 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.32 | 0.08 | 0.12 | 103 |
| Corn___Common_rust | 0.98 | 0.99 | 0.98 | 238 |
| Corn___Northern_Leaf_Blight | 0.39 | 0.48 | 0.43 | 197 |
| Corn___healthy | 0.83 | 0.86 | 0.85 | 231 |
| Grape___Black_rot | 0.26 | 0.23 | 0.24 | 236 |
| Grape___Esca_(Black_Measles) | 0.52 | 0.56 | 0.54 | 277 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.26 | 0.21 | 0.23 | 215 |
| Grape___healthy | 0.50 | 0.32 | 0.39 | 85 |
| Orange___Haunglongbing_(Citrus_greening) | 0.54 | 0.61 | 0.57 | 1,102 |
| Peach___Bacterial_spot | 0.31 | 0.29 | 0.30 | 460 |
| Peach___healthy | 0.25 | 0.24 | 0.24 | 72 |
| Pepper,_bell___Bacterial_spot | 0.37 | 0.28 | 0.32 | 199 |
| Pepper,_bell___healthy | 0.28 | 0.14 | 0.18 | 295 |
| Potato___Early_blight | 0.43 | 0.62 | 0.51 | 200 |
| Potato___Late_blight | 0.17 | 0.09 | 0.11 | 200 |
| Potato___healthy | 0.00 | 0.00 | 0.00 | 30 |
| Raspberry___healthy | 0.53 | 0.22 | 0.31 | 74 |
| Soybean___healthy | 0.64 | 0.86 | 0.73 | 1,018 |
| Squash___Powdery_mildew | 0.40 | 0.48 | 0.44 | 367 |
| Strawberry___Leaf_scorch | 0.33 | 0.37 | 0.35 | 222 |
| Strawberry___healthy | 0.26 | 0.15 | 0.19 | 91 |
| Tomato___Bacterial_spot | 0.50 | 0.45 | 0.47 | 426 |
| Tomato___Early_blight | 0.27 | 0.17 | 0.21 | 200 |
| Tomato___Late_blight | 0.36 | 0.31 | 0.33 | 381 |
| Tomato___Leaf_Mold | 0.30 | 0.19 | 0.24 | 190 |
| Tomato___Septoria_leaf_spot | 0.35 | 0.31 | 0.33 | 354 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.32 | 0.35 | 0.33 | 335 |
| Tomato___Target_Spot | 0.28 | 0.25 | 0.26 | 281 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.34 | 0.48 | 0.40 | 1,072 |
| Tomato___Tomato_mosaic_virus | 0.26 | 0.20 | 0.23 | 75 |
| Tomato___healthy | 0.42 | 0.58 | 0.49 | 318 |
| **Accuracy** | | | **0.44** | **10,859** |
| **Macro avg** | **0.38** | **0.34** | **0.35** | **10,859** |
| **Weighted avg** | **0.42** | **0.44** | **0.43** | **10,859** |

#### Model Size

- `leaf_disease_model.pkl` (original, no compression)
- `leaf_disease_model_CLEAN.pkl` (compress=3): **699.76 MB**
- `leaf_disease_model_TINY.pkl` (zlib level 9): **~310 MB**

#### Output Files

- `leaf_disease_model_CLEAN.pkl` (compress=3, ~700 MB, Git LFS)
- `leaf_disease_model_TINY.pkl` (heavy compression, Git LFS)
- `confusion_matrix.png` (12x10 inches, 150 DPI, Blues colormap)
- `feature_importance.png` (8x5 inches, 150 DPI, green horizontal bars)
- `classification_report.txt`

#### Why 44%

Only 7 grayscale features for 38 classes. No color information (RGB), no texture information (HOG). Many diseases look identical in grayscale statistics alone — the model has almost nothing to distinguish them. Corn___Common_rust is an outlier at 98/99 because its distinctive rust-orange color creates unique grayscale intensity patterns.

---

### T — Experiment 2: HOG + PCA + RF (Broken)

**File:** `T/exp2/sage_pca.ipynb`
**Platform:** AWS SageMaker Studio (ml.c5.4xlarge — 16 vCPU, 32 GB RAM)

#### Preprocessing

1. **Gaussian Blur:** `cv2.GaussianBlur(image, (5,5), 0)` — reduces background noise
2. **Resize:** 128x128 pixels
3. **Grayscale conversion:** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
4. **Histogram:** 256-bin, normalized to probability distribution

#### Features Extracted (8,113 per image)

**13 Color + Grayscale Features:**

| # | Feature | Source |
|---|---------|--------|
| 1 | Red channel mean | RGB |
| 2 | Green channel mean | RGB |
| 3 | Blue channel mean | RGB |
| 4 | Red channel std | RGB |
| 5 | Green channel std | RGB |
| 6 | Blue channel std | RGB |
| 7 | Grayscale mean | Gray |
| 8 | Grayscale std | Gray |
| 9 | Grayscale min | Gray |
| 10 | Grayscale max | Gray |
| 11 | Grayscale median | Gray |
| 12 | Grayscale variance | Gray |
| 13 | Grayscale entropy | Gray |

**~8,100 HOG (Histogram of Oriented Gradients) Features:**

| Parameter | Value |
|-----------|-------|
| orientations | 9 |
| pixels_per_cell | **(8, 8)** |
| cells_per_block | (2, 2) |
| block_norm | L2-Hys |
| Input image | 128x128 grayscale |
| Output features | **~8,100** |

**Total raw features:** 13 + 8,100 = **8,113**

#### Dimensionality Reduction (PCA)

| Step | Detail |
|------|--------|
| StandardScaler | Applied to all 8,113 features (zero mean, unit variance) |
| PCA | `n_components=0.95` (retain 95% variance) |
| Features after PCA | **2,074** (compressed from 8,113) |

#### Model Configuration

- **max_depth:** 25
- **min_samples_leaf:** 2
- Everything else: common configuration

#### Train/Test Split

- Training: 43,434 / Test: 10,859

#### Results — Accuracy: 53.55%

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Corn___Common_rust (best) | 0.88 | 0.94 | 0.91 | 238 |
| Grape___Leaf_blight | 0.71 | 0.78 | 0.74 | 215 |
| Peach___healthy | 1.00 | 0.06 | 0.11 | 72 |
| Pepper,_bell___Bacterial_spot | 1.00 | 0.01 | 0.02 | 199 |
| Apple___Apple_scab | 0.00 | 0.00 | 0.00 | 126 |
| Apple___Cedar_apple_rust | 0.00 | 0.00 | 0.00 | 55 |
| Potato___healthy | 0.00 | 0.00 | 0.00 | 30 |
| Tomato___Tomato_mosaic_virus | 0.00 | 0.00 | 0.00 | 75 |
| **Macro avg** | **0.55** | **0.39** | — | **10,859** |

4 classes had exactly 0% precision and 0% recall (never predicted at all).

#### Saved Pipeline

```python
pipeline = {
    "scaler": StandardScaler,
    "pca": PCA(n_components=0.95),
    "model": RandomForestClassifier,
    "classes": 38 class names
}
```
File: `leaf_disease_pipeline.pkl`

#### Why 53% — The PCA + Random Forest Mismatch (Root Cause Analysis)

**Root cause:** PCA rotates features diagonally to compress them. Random Forest makes axis-aligned splits (horizontal/vertical decision boundaries like "if feature_X > threshold"). When PCA-rotated features are fed to a Random Forest, the tree cannot draw clean splits — it must approximate diagonal boundaries with hundreds of tiny "staircase" splits. With `max_depth=25`, it runs out of depth before completing the staircase.

**Evidence:**
- Classes with distinctive single-feature signals survived PCA partially (Corn_Common_rust: 94% recall — its strong orange color projects onto PCA components in a partially axis-separable way)
- Classes depending on subtle multi-feature patterns collapsed to 0-2% recall (Apple_Scab, Pepper_Bell_Bacterial_spot, Tomato_Early_blight)
- 4 classes had exactly 0% precision and 0% recall (never predicted at all)

**Rule:** PCA is compatible with SVMs, logistic regression, and neural networks (which can learn rotated boundaries). PCA is **NOT** compatible with tree-based models (Random Forest, Gradient Boosted Trees, Decision Trees) which rely on axis-aligned splits.

---

### T — Experiment 3a: HOG Without PCA, depth=25 (Fixed)

**File:** `T/exp3/a/sage_nopca.ipynb`
**Platform:** AWS SageMaker Studio (ml.c5.4xlarge — 16 vCPU, 32 GB RAM)
**Also available as:** `sage_nopca_full.ipynb` (includes dataset download), `colab_nopca.ipynb` and `colab_nopca.py` (Google Colab versions)

#### Preprocessing

Identical to Experiment 2: Gaussian Blur (5x5) -> Resize 128x128 -> Grayscale -> 256-bin histogram

#### Features Extracted (1,777 per image)

**13 Color + Grayscale Features:** Same as Experiment 2.

**~1,764 HOG Features:**

| Parameter | Value |
|-----------|-------|
| orientations | 9 |
| pixels_per_cell | **(16, 16)** (changed from 8x8) |
| cells_per_block | (2, 2) |
| block_norm | L2-Hys |
| Input image | 128x128 grayscale |
| Output features | **~1,764** (down from ~8,100) |

**Total features:** 13 + 1,764 = **1,777**

#### Key Changes from Experiment 2

| What | Experiment 2 (PCA) | Experiment 3 (No PCA) |
|------|--------------------|-----------------------|
| HOG pixels_per_cell | (8, 8) -> 8,100 features | (16, 16) -> 1,764 features |
| PCA | Yes (8,113 -> 2,074 rotated) | **Removed** |
| StandardScaler | Yes | **Removed** |
| Features to RF | 2,074 PCA-rotated | 1,777 raw unrotated |
| Pipeline saved | scaler + pca + model | model only |

**Why (16,16) instead of (8,8):** Increasing `pixels_per_cell` from (8,8) to (16,16) naturally reduces HOG output from ~8,100 to ~1,764 features — keeping the model small without needing PCA compression. The raw, unrotated features can be split cleanly by the Random Forest's axis-aligned decision boundaries.

#### Model Configuration

- **max_depth:** 25
- **min_samples_leaf:** 2
- Everything else: common configuration

#### Results — Accuracy: 75.52%

**Full 38-class classification report:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Apple___Apple_scab | 0.93 | 0.10 | 0.19 | 126 |
| Apple___Black_rot | 0.78 | 0.83 | 0.80 | 124 |
| Apple___Cedar_apple_rust | 0.00 | 0.00 | 0.00 | 55 |
| Apple___healthy | 0.60 | 0.42 | 0.49 | 329 |
| Blueberry___healthy | 0.71 | 0.85 | 0.77 | 300 |
| Cherry___Powdery_mildew | 0.89 | 0.75 | 0.81 | 210 |
| Cherry___healthy | 0.93 | 0.86 | 0.89 | 171 |
| Corn___Cercospora_leaf_spot | 0.80 | 0.36 | 0.50 | 103 |
| Corn___Common_rust | 0.97 | 0.99 | 0.98 | 238 |
| Corn___Northern_Leaf_Blight | 0.65 | 0.80 | 0.72 | 197 |
| Corn___healthy | 0.94 | 0.90 | 0.92 | 231 |
| Grape___Black_rot | 0.80 | 0.60 | 0.69 | 236 |
| Grape___Esca_(Black_Measles) | 0.68 | 0.91 | 0.78 | 277 |
| Grape___Leaf_blight | 0.94 | 0.96 | 0.95 | 215 |
| Grape___healthy | 0.99 | 0.89 | 0.94 | 85 |
| Orange___Haunglongbing | 0.84 | 0.95 | 0.89 | 1,102 |
| Peach___Bacterial_spot | 0.65 | 0.76 | 0.70 | 460 |
| Peach___healthy | 0.74 | 0.62 | 0.68 | 72 |
| Pepper,_bell___Bacterial_spot | 0.65 | 0.39 | 0.49 | 199 |
| Pepper,_bell___healthy | 0.84 | 0.48 | 0.62 | 295 |
| Potato___Early_blight | 0.78 | 0.81 | 0.80 | 200 |
| Potato___Late_blight | 0.83 | 0.43 | 0.57 | 200 |
| Potato___healthy | 0.00 | 0.00 | 0.00 | 30 |
| Raspberry___healthy | 1.00 | 0.61 | 0.76 | 74 |
| Soybean___healthy | 0.70 | 0.97 | 0.81 | 1,018 |
| Squash___Powdery_mildew | 0.75 | 0.93 | 0.83 | 367 |
| Strawberry___Leaf_scorch | 0.80 | 0.86 | 0.83 | 222 |
| Strawberry___healthy | 0.92 | 0.62 | 0.74 | 91 |
| Tomato___Bacterial_spot | 0.72 | 0.77 | 0.74 | 426 |
| Tomato___Early_blight | 0.58 | 0.22 | 0.32 | 200 |
| Tomato___Late_blight | 0.86 | 0.49 | 0.62 | 381 |
| Tomato___Leaf_Mold | 0.69 | 0.57 | 0.62 | 190 |
| Tomato___Septoria_leaf_spot | 0.78 | 0.37 | 0.50 | 354 |
| Tomato___Spider_mites | 0.66 | 0.83 | 0.74 | 335 |
| Tomato___Target_Spot | 0.62 | 0.65 | 0.64 | 281 |
| Tomato___Yellow_Leaf_Curl_Virus | 0.70 | 0.89 | 0.79 | 1,072 |
| Tomato___Tomato_mosaic_virus | 0.84 | 0.43 | 0.57 | 75 |
| Tomato___healthy | 0.83 | 0.92 | 0.87 | 318 |
| **Accuracy** | | | **0.76** | **10,859** |
| **Weighted avg** | **0.76** | **0.76** | **0.74** | **10,859** |
| **Macro avg** | **0.75** | **0.65** | **0.67** | **10,859** |

#### Model Size

- `leaf_disease_pipeline_v2.pkl` (compress=3): **445.6 MB**

#### Saved Pipeline

```python
pipeline = {
    "model": RandomForestClassifier,
    "classes": 38 class names
}
```
No scaler or PCA saved — not needed.

#### Remaining Bottleneck

`max_depth=25` caps trees from fully learning harder classes. Two classes at 0% recall: Apple___Cedar_apple_rust (55 test samples) and Potato___healthy (30 test samples) — extreme class imbalance.

---

### T — Experiment 3b: HOG Without PCA, depth=None (Feature Ceiling Test)

**File:** `T/exp3/b/sage_nopca_bestresults.ipynb`
**Platform:** AWS SageMaker Studio (ml.c5.4xlarge — 16 vCPU, 32 GB RAM)

#### Configuration

Same as Experiment 3a except:
- **max_depth:** None (unlimited tree growth)
- Everything else identical (500 trees, balanced, random_state=42, 1,777 raw features)

#### Results — Accuracy: 75.31%

**Full 38-class classification report:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Apple___Apple_scab | 0.70 | 0.06 | 0.10 | 126 |
| Apple___Black_rot | 0.82 | 0.81 | 0.81 | 124 |
| Apple___Cedar_apple_rust | 0.00 | 0.00 | 0.00 | 55 |
| Apple___healthy | 0.65 | 0.41 | 0.50 | 329 |
| Blueberry___healthy | 0.74 | 0.86 | 0.80 | 300 |
| Cherry___Powdery_mildew | 0.91 | 0.73 | 0.81 | 210 |
| Cherry___healthy | 0.92 | 0.85 | 0.88 | 171 |
| Corn___Cercospora_leaf_spot | 0.85 | 0.34 | 0.49 | 103 |
| Corn___Common_rust | 0.97 | 0.99 | 0.98 | 238 |
| Corn___Northern_Leaf_Blight | 0.65 | 0.81 | 0.72 | 197 |
| Corn___healthy | 0.94 | 0.90 | 0.92 | 231 |
| Grape___Black_rot | 0.82 | 0.62 | 0.71 | 236 |
| Grape___Esca_(Black_Measles) | 0.69 | 0.91 | 0.78 | 277 |
| Grape___Leaf_blight | 0.94 | 0.94 | 0.94 | 215 |
| Grape___healthy | 1.00 | 0.85 | 0.92 | 85 |
| Orange___Haunglongbing | 0.85 | 0.95 | 0.90 | 1,102 |
| Peach___Bacterial_spot | 0.66 | 0.77 | 0.71 | 460 |
| Peach___healthy | 0.87 | 0.67 | 0.76 | 72 |
| Pepper,_bell___Bacterial_spot | 0.66 | 0.41 | 0.51 | 199 |
| Pepper,_bell___healthy | 0.85 | 0.50 | 0.63 | 295 |
| Potato___Early_blight | 0.82 | 0.80 | 0.81 | 200 |
| Potato___Late_blight | 0.79 | 0.39 | 0.52 | 200 |
| Potato___healthy | 0.00 | 0.00 | 0.00 | 30 |
| Raspberry___healthy | 1.00 | 0.57 | 0.72 | 74 |
| Soybean___healthy | 0.69 | 0.98 | 0.81 | 1,018 |
| Squash___Powdery_mildew | 0.75 | 0.94 | 0.83 | 367 |
| Strawberry___Leaf_scorch | 0.78 | 0.87 | 0.82 | 222 |
| Strawberry___healthy | 0.93 | 0.55 | 0.69 | 91 |
| Tomato___Bacterial_spot | 0.75 | 0.77 | 0.76 | 426 |
| Tomato___Early_blight | 0.64 | 0.17 | 0.27 | 200 |
| Tomato___Late_blight | 0.87 | 0.49 | 0.63 | 381 |
| Tomato___Leaf_Mold | 0.68 | 0.54 | 0.60 | 190 |
| Tomato___Septoria_leaf_spot | 0.75 | 0.35 | 0.48 | 354 |
| Tomato___Spider_mites | 0.68 | 0.81 | 0.74 | 335 |
| Tomato___Target_Spot | 0.66 | 0.62 | 0.63 | 281 |
| Tomato___Yellow_Leaf_Curl_Virus | 0.64 | 0.92 | 0.76 | 1,072 |
| Tomato___Tomato_mosaic_virus | 0.86 | 0.33 | 0.48 | 75 |
| Tomato___healthy | 0.84 | 0.91 | 0.87 | 318 |
| **Accuracy** | | | **0.75** | **10,859** |
| **Weighted avg** | **0.76** | **0.75** | **0.73** | **10,859** |
| **Macro avg** | **0.75** | **0.64** | **0.67** | **10,859** |

#### Model Size

- `leaf_disease_pipeline_best.pkl` (compress=3): **511.1 MB**

#### Why depth=None Didn't Help

Removing the depth cap gave 75.31% — essentially identical to 75.52% with depth=25. The model grew from 445.6 MB to 511.1 MB (+65.5 MB) with **no accuracy gain**. This proves:

1. **max_depth was NOT the bottleneck.** The trees at depth=25 already captured everything the features could tell them.
2. **~75% is the hard feature ceiling** for HOG (16,16) + 13 color/grayscale stats on 38 classes.
3. **Deeper trees just overfit** — they memorize training data without learning better separating patterns.
4. Apple___Cedar_apple_rust (55 samples) and Potato___healthy (30 samples) stayed at 0% recall — even unlimited depth couldn't help with extreme class imbalance.
5. Apple___Apple_scab dropped from 10% to 6% recall with depth=None — deeper trees overfit on this small class.

This validates the need for deep learning (Experiment 4): only learned features (CNN) can push beyond the ~75% ceiling.

---

### T — Experiment 4: MobileNetV3-Small (Completed)

**File:** `T/exp4/sage_mobilenetresult.ipynb`
**Platform:** AWS SageMaker (ml.g4dn.xlarge — 1x NVIDIA T4 GPU)

#### Approach & Fixes Applied

- **Architecture:** MobileNetV3-Small (pre-trained on ImageNet)
- **Training Strategy:** Two-phase transfer learning (Phase 1: freeze base, train head; Phase 2: unfreeze top layers, fine-tune at lower learning rate)
- **Data Augmentation:** Implemented `RandomFlip`, `RandomRotation`, and `RandomZoom` to prevent overfitting.
- **Input Size:** 224x224x3 (`IMG_SIZE = 224` in the actual training notebook). MobileNetV3 handles normalization internally and expects `[0, 255]` pixel inputs — the original `img / 255.0` normalization was removed, which was critical for preserving the pre-trained weights.
- **Learning Rate:** Phase 2 (fine-tuning) used a specialized lower learning rate (`1e-5` / `1e-4`) to ensure stable training.

#### Results

- **Accuracy:** **98.13%** (massive improvement from the 75.52% feature ceiling of previous traditional ML experiments)
- **Model Sizes:** `mobilenet_leaf_disease.h5` is 8.7 MB (Keras version), while the quantized for edge deployment `mobilenet_leaf_disease.tflite` is just **1.1 MB**.
- **Performance:** Excellent precision and recall across all 38 classes (mostly > 0.93 per class).

#### Status

Experiment successfully executed. The `.tflite`, `.h5`, and generated plots (`confusion_matrix_mobilenet.png`, `training_curves_mobilenet.png`) are available.

---

## Accuracy Progression (T's 38-Class Experiments)

| Exp | Method | Features | Accuracy | Model Size | Platform |
|-----|--------|----------|----------|------------|----------|
| 1 | 7 grayscale stats -> RF (depth=None) | 7 | **44.48%** | 699.8 MB | Google Colab (CPU) |
| 2 | 13 color + HOG 8x8 -> PCA -> RF (depth=25) | 2,074 (rotated) | **53.55%** | -- | SageMaker ml.c5.4xlarge |
| 3a | 13 color + HOG 16x16 -> RF (depth=25) | 1,777 (raw) | **75.52%** | 445.6 MB | SageMaker ml.c5.4xlarge |
| 3b | 13 color + HOG 16x16 -> RF (depth=None) | 1,777 (raw) | **75.31%** | 511.1 MB | SageMaker ml.c5.4xlarge |
| 4 | MobileNetV3-Small (transfer learning) | Learned by CNN | **98.13%** | 1.1 MB (.tflite) | SageMaker ml.g4dn.xlarge |

### Key Research Findings

1. **7 grayscale features are insufficient for 38 classes** (44.48%). Many diseases look identical in grayscale statistics. Corn___Common_rust is an exception (98/99% P/R) because its distinctive color creates unique grayscale patterns.
2. **Adding color (RGB) and texture (HOG) features** improved accuracy from 44% to 53% (Exp 1 -> 2), but PCA suppressed most of the gain.
3. **PCA + Random Forest is incompatible.** PCA rotates features diagonally; RF splits axis-aligned. The mismatch forces inefficient staircase approximations that waste tree depth. Removing PCA gave a **22-point accuracy jump** (53% -> 76%).
4. **~75% is the hard feature ceiling for HOG + RF on 38 classes.** Removing the depth cap (3a -> 3b) gave zero improvement (−0.21% with 65 MB more model size), proving the bottleneck is feature quality, not model capacity.
5. **Class imbalance is persistent.** Potato___healthy (152 total, 30 test) and Apple___Cedar_apple_rust (275 total, 55 test) remained at 0% recall across ALL experiments — Exp 1, 2, 3a, and 3b — regardless of method, features, or tree depth.
6. **Only learned features (CNN) can push beyond ~75%.** This validates Experiment 4 (MobileNetV3-Small).

---

## Why Hybrid CNN+RF Was Rejected (Appendix)

A fifth experiment was considered: use MobileNetV3 as a feature extractor (remove its classification head) and feed the CNN's learned features into a Random Forest.

```
Hybrid: Image -> [MobileNetV3 body (frozen)] -> 1,024 deep features -> Random Forest -> Prediction
End-to-end: Image -> [MobileNetV3 full] -> Dense layer (38 outputs) -> Prediction
```

### The Appeal

CNN-level feature quality + RF data efficiency + CPU inference.

| Property | HOG + RF (Exp 1-3) | MobileNet E2E (Exp 4) | Hybrid CNN+RF |
|----------|-------------------|----------------------|---------------|
| Who designs features? | You (manually) | CNN (automatically) | CNN (automatically) |
| Who classifies? | Random Forest | Dense layer | Random Forest |
| Feature quality | Limited by HOG | Excellent | Excellent |
| Needs GPU for training? | No | Yes | Only for one-time feature extraction |
| Needs GPU for inference? | No | Preferred | No (RF on CPU) |

### Problem 1: The "Two Brains" Software Dependency

**End-to-end (Exp 4):** Export one `.tflite` file. Mobile developer loads it with TFLite library. One framework, one file, works on phone's NPU/GPU natively.

**Hybrid:** Load `.tflite` to run MobileNet body -> get 1,024 numbers. Then load `rf_classifier.pkl` and run RF. On Android/iOS this requires both TFLite AND a scikit-learn equivalent. There is no official scikit-learn runtime for mobile. The developer would have to port the RF to C++ manually, bundle a Python interpreter, or convert the RF to a custom if-else engine.

**For Edge AI, single-framework deployment is critical.** Farmers' phones don't have Python. Drones run C firmware. Raspberry Pi deployments want minimal dependencies.

### Problem 2: The Mathematical File Size Irony

In Experiments 1-3, the RF IS the entire model — it makes sense that it's large (445-700 MB).

In the hybrid, the MobileNetV3 classification layer being replaced is:

```
Dense layer: 1,024 inputs x 38 outputs = 38,912 weights + 38 biases = 38,950 parameters
At 4 bytes each = 155.8 KB
```

A Random Forest on 1,024 deep features with 500 trees would weigh **10-20 MB**. So the hybrid replaces a 155 KB classification layer with a 10-20 MB RF — **100x larger** at the classification step for no accuracy gain.

| Component | End-to-End (Exp 4) | Hybrid |
|-----------|-------------------|--------|
| MobileNetV3 body | ~3.4 MB | ~3.4 MB |
| Classification layer | ~155 KB (Dense) | ~10-20 MB (Random Forest) |
| **Total** | **~3.5 MB** | **~15-25 MB** |
| Extra framework needed | None | scikit-learn runtime |

### Problem 3: The RF Was Never the Bottleneck

Experiments 1-3 proved the RF classified correctly when given good features (Corn_Common_rust: 99% recall from HOG alone). It failed when HOG features couldn't capture the visual difference. Once features are good enough (CNN features), even a 155 KB Dense layer classifies correctly. No need for a 10-20 MB RF when the problem was never the classifier — it was the features.

### Conclusion

The Random Forest is a great classifier when it IS the entire model (Experiments 1-3). There's nothing else doing the work — it handles both feature interpretation and classification, so its large size (445-700 MB) is justified. But once you're already running a neural network for feature extraction, the Dense layer is smaller (155 KB vs 10-20 MB), faster, and requires no extra software framework. The hybrid approach adds software complexity, increases model size, and solves a problem that doesn't exist — the RF was never the weak link, the features were.

End-to-end MobileNetV3-Small provides optimal Edge AI deployment: 98.13% accuracy, 1.1 MB .tflite, single framework, no extra dependencies.

#### Paper Discussion Section Quote (Ready to Use)

> "While a hybrid architecture (CNN feature extractor + Random Forest classifier) was evaluated conceptually, it violates Edge AI deployment principles. The 'two brains' software dependency — requiring both TFLite and a scikit-learn runtime on the target device — introduces integration complexity incompatible with resource-constrained field deployment. Furthermore, replacing the final Dense layer (155 KB, 38,950 parameters) with a Random Forest (10-20 MB) actually increases the model's edge footprint by two orders of magnitude at the classification stage. Therefore, end-to-end MobileNetV3-Small provides the optimal balance of 98.13% accuracy within a single 1.1 MB TFLite package suitable for direct phone, drone, or Raspberry Pi deployment."

---

## Comparison & Analysis: Vishnu vs T

Both team members worked on crop leaf disease detection using the PlantVillage dataset, but with fundamentally different scopes and approaches.

### Overview

| Aspect | Vishnu | T |
|--------|--------|-------|
| **Dataset source** | `emmarex/plantdisease` (folder format) | `sebastianpalaciob/plantvillage-for-object-detection-yolo` (YOLO) |
| **Classes** | 5 (Potato + Tomato) | 38 (all diseases) |
| **Total images** | 5,652 | 54,293 |
| **Platform** | Google Colab (free CPU) | Colab (Exp 1), SageMaker ml.c5.4xlarge (Exp 2-3), ml.g4dn.xlarge (Exp 4) |
| **Experiments** | 1 approach (GB + RF) | 4 progressive experiments |
| **Best accuracy** | 83.90% (RF, 5 classes) | 98.13% (MobileNetV3, 38 classes) |
| **Deployment** | Gradio on HuggingFace Spaces (live) | Edge AI (paper focus, not yet deployed) |
| **Goal** | Working web app prototype | Research paper with ablation study |
| **Model format** | `.pkl` (scikit-learn) | `.pkl` (Exp 1-3), `.tflite` (Exp 4) |
| **Production-ready?** | Yes (live on HuggingFace) | Not yet (experiments phase) |

### Direct Comparison: Dataset Scope

| | Vishnu | T |
|---|---|---|
| Dataset source | `emmarex/plantdisease` (folder format) | `sebastianpalaciob/plantvillage-for-object-detection-yolo` (YOLO format) |
| Total classes | 5 | 38 |
| Total images | 5,652 | 54,293 |
| Class imbalance | Mild (152 to 1,909 — 12.6:1 ratio) | Severe (152 to 5,507 — 36:1 ratio) |
| Plant species | 2 (Potato, Tomato) | 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato) |

**Why the accuracy numbers are not directly comparable:** T's 38-class problem is fundamentally harder than Vishnu's 5-class problem. With 38 classes, diseases across different plant species can look visually similar (e.g., early blight on potato vs tomato), and the severe class imbalance (Potato___healthy has only 152 samples vs Orange___Haunglongbing with 5,507) makes minority-class classification extremely difficult. Vishnu's 83.90% on 5 classes and T's 75.52% on 38 classes cannot be compared as raw numbers — the 38-class task has 7.6x more classes to distinguish, 9.6x more images to process, and far more confusing inter-class boundaries.

### Direct Comparison: Feature Engineering

| | Vishnu | T |
|---|---|---|
| Color features | RGB mean + RGB std (6) | RGB mean + RGB std (6) — Exp 2-3 only |
| Grayscale features | Texture std (1) | Mean, std, min, max, median, variance, entropy (7) |
| HOG features | Defined in code but never used in training | Used in Exp 2-3 (1,764 features at 16x16 cells) |
| Total features | 7 | 7 (Exp 1) -> 2,074 rotated (Exp 2) -> 1,777 raw (Exp 3) |
| PCA | Yes (on 7 features for GB model, reduced to 2) | Tried in Exp 2 (8,113->2,074) and removed in Exp 3 (caused 22% accuracy drop) |
| Gaussian blur | Yes (5,5) | No (Exp 1), Yes (5,5) (Exp 2-3) |

Both started with similar simple statistical features. T progressively added HOG texture features and discovered that PCA is incompatible with Random Forest through empirical experimentation — a finding that applies to both projects.

### Direct Comparison: Model & Deployment

| | Vishnu | T |
|---|---|---|
| Classifiers tried | Gradient Boosting + Random Forest | Random Forest (Exp 1-3), MobileNetV3 (Exp 4) |
| Final classifier | Random Forest (500 trees, balanced) | MobileNetV3-Small (Exp 4, 98.13%) |
| Deployment target | Web app (Gradio on HuggingFace Spaces) | Edge devices (phone, drone, Raspberry Pi) |
| Model format | `.pkl` (scikit-learn, joblib) | `.pkl` (Exp 1-3), `.tflite` (Exp 4) |
| Model size | Single model file | 445-700 MB (RF), 1.1 MB (.tflite) |
| Production-ready? | Yes (live on HuggingFace) | Not yet (paper experiments phase) |
| Inference framework | Python (scikit-learn + Gradio) | Python (Exp 1-3), TFLite (Exp 4) |

### Feature Engineering Comparison (Full Detail)

| | Vishnu (Training) | Vishnu (Deployed app.py) | T (Exp 1) | T (Exp 2) | T (Exp 3) |
|---|---|---|---|---|---|
| **Color features** | RGB mean (3) + RGB std (3) | None | None | RGB mean (3) + RGB std (3) | RGB mean (3) + RGB std (3) |
| **Grayscale features** | Texture std (1) | Mean, std, min, max, median, variance, entropy (7) | Mean, std, min, max, median, variance, entropy (7) | Mean, std, min, max, median, variance, entropy (7) | Mean, std, min, max, median, variance, entropy (7) |
| **HOG features** | Defined but unused | None | None | ~8,100 (8x8 cells) | ~1,764 (16x16 cells) |
| **Gaussian blur** | Yes (5,5) | No | No | Yes (5,5) | Yes (5,5) |
| **PCA** | Used for GB model only | No | No | Yes (8,113->2,074) | **Removed** |
| **StandardScaler** | Used for GB model only | No | No | Yes | **Removed** |
| **Total features** | **7** | **7** | **7** | **2,074 (rotated)** | **1,777 (raw)** |

### Vishnu's Code Structure & What Was Delivered

Vishnu's work follows a linear pipeline from data to deployment:

1. **Dataset download** from Kaggle (`emmarex/plantdisease`) — 658 MB
2. **5-class subset extraction** into `dataset/` folder (Tomato + Potato diseases)
3. **Feature extraction** — 7 color features per image -> saved to `leaf_features.csv`
4. **Two models trained:**
   - Gradient Boosting + PCA -> 62.86% (dropped Potato___healthy to 0%)
   - Random Forest (500 trees, balanced) -> 83.90% (final model)
5. **Model saved** as `leaf_disease_model.pkl`
6. **Gradio web app** deployed on Hugging Face Spaces (live)

**Key observation:** The deployed `app.py` uses a **different feature set** (7 grayscale stats: mean, std, min, max, median, variance, entropy) than the training code (7 color features: RGB mean, RGB std, gray texture std). The `project_code.py` also defines a HOG extraction function that is never called during training. This means the deployed model receives features from a different feature space than it was trained on.

**What was delivered:** A working end-to-end prototype — from raw Kaggle data to a live web application. The app is accessible at https://huggingface.co/spaces/chinni-1221/crop-leaf-disease-detection and accepts leaf image uploads, returning a disease classification label.

### T's Experiment Arc & What Was Discovered

T's work follows a progressive experiment design, where each experiment builds on the failure analysis of the previous one:

1. **Exp 1 (Baseline):** 7 grayscale features -> 44.48%. Conclusion: grayscale alone is insufficient for 38 classes.
2. **Exp 2 (Added HOG+PCA):** 8,113 features compressed to 2,074 via PCA -> 53.55%. Only +9 points despite adding thousands of texture features. Root cause diagnosed: PCA rotates features diagonally, making them incompatible with RF's axis-aligned splits.
3. **Exp 3a (Removed PCA):** 1,777 raw features -> 75.52%. **+22 point jump** from a single algorithmic fix (removing PCA). Proved the PCA diagnosis correct.
4. **Exp 3b (Removed depth cap):** Same features, depth=None -> 75.31%. Zero improvement with a 65.5 MB larger model. Proved ~75% is the **hard feature ceiling** — the bottleneck is feature quality, not model capacity.
5. **Exp 4 (MobileNetV3, completed):** End-to-end deep learning pushed beyond the feature ceiling. Achieved **98.13% accuracy** with a 1.1 MB .tflite model.

**Key discoveries:**
- PCA + Random Forest is a known mathematical incompatibility (PCA rotates, RF splits axis-aligned)
- ~75% is the hard accuracy limit of hand-crafted features (HOG + color stats) on 38 PlantVillage classes
- Deeper trees do not help when features are the bottleneck (overfitting without accuracy gain)
- Potato___healthy (152 samples) and Apple___Cedar_apple_rust (275 samples) have 0% recall across ALL experiments — a persistent class imbalance problem

**What was delivered:** A complete ablation study suitable for a research paper, demonstrating why traditional ML plateaus and why deep learning (CNN) is necessary for Edge AI deployment on this dataset.

### Contributions

**Vishnu:** Built a working end-to-end prototype — data download, feature extraction, training, and live deployment on Hugging Face Spaces. Focused on delivering a usable product for 5 common Potato/Tomato diseases. Demonstrated the practical engineering pipeline from raw data to web app.

**T:** Deep technical investigation — diagnosed PCA+RF incompatibility, proved the 75% feature ceiling through ablation experiments (Exp 3a vs 3b), analyzed and rejected the Hybrid CNN+RF approach, and designed a progressive experiment arc for a research paper. Focused on understanding *why* traditional ML fails and demonstrating the need for deep learning on edge devices.

### How They Complement Each Other

Vishnu's work provides the **deployment pipeline** (Gradio app, HuggingFace hosting, end-to-end usability), while T's work provides the **research depth** (why certain approaches fail, what the accuracy limits are, and how deep learning overcomes them). Together they cover both the practical engineering and the scientific analysis needed for a complete project.

For the research paper: Vishnu's work demonstrates that the system can be deployed as a web application, while T's experiments provide the scientific justification for choosing MobileNetV3 over traditional ML — with ablation evidence showing the progressive failure of simpler approaches and the ~75% ceiling that only CNN features can break through.

---

## Directory Structure

```
crop/
|-- README.md                             # This file (complete project documentation)
|-- .gitattributes                        # Git LFS: *.pkl tracked
|-- .gitignore                            # Ignores __pycache__, .ipynb_checkpoints, kaggle.json, .cursor, .claude, benchmark_inference*.ipynb
|
|-- vishnu/
|   |-- README.md                         # Vishnu's complete technical README
|   |-- project_code.py                   # Full Colab training pipeline (.py with cell separators)
|   |-- Project.ipynb                     # Jupyter notebook with all cell outputs
|   |-- app.py                            # Deployed Gradio web app (7 grayscale features)
|   |-- leaf_disease_model.pkl            # Trained RF model (500 trees, 5 classes, Git LFS)
|   |-- requirements (1).txt              # Deployment dependencies (gradio, opencv, sklearn, etc.)
|   |-- gitattributes                     # HuggingFace LFS config (25+ binary extensions)
|
|-- T/
|   |-- exp1/                             # Experiment 1: 7-Feature Baseline (44.48%)
|   |   |-- Crop.ipynb                    # Google Colab notebook (7 grayscale, RF, 44.48%)
|   |   |-- crop_code.py                  # Same code as .py script for Colab
|   |   |-- train_model.py                # Standalone local script (expects class subfolders)
|   |   |-- classification_report.txt     # Full 38-class precision/recall/F1
|   |   |-- confusion_matrix.png          # 38x38 confusion matrix (12x10in, 150 DPI, Blues)
|   |   |-- feature_importance.png        # 7-feature importance bar chart (8x5in, green bars)
|   |   |-- leaf_disease_model_CLEAN.pkl  # Trained model (compress=3, 699.76 MB, Git LFS)
|   |   |-- leaf_disease_model_TINY.pkl   # Heavy compression version (~310 MB, Git LFS)
|   |   |-- exp1_methodology.md           # Step-by-step methodology with pixel examples
|   |
|   |-- exp2/                             # Experiment 2: HOG+PCA (53.55%, broken)
|   |   |-- sage_pca.ipynb                # SageMaker notebook with outputs (HOG 8x8, PCA, RF depth=25)
|   |
|   |-- exp3/                             # Experiment 3: HOG No PCA (75.52% / 75.31%)
|   |   |-- a/                            # Experiment 3a: depth=25 (75.52%)
|   |   |   |-- sage_nopca.ipynb          # Notebook with dataset download
|   |   |   |-- sage_nopca_full.ipynb     # Notebook with full dataset setup
|   |   |   |-- colab_nopca.ipynb         # Colab alternative
|   |   |   |-- colab_nopca.py            # Colab alternative
|   |   |   |-- leaf_disease_pipeline_v2.pkl
|   |   |   |-- classification_report_v2.txt
|   |   |   |-- confusion_matrix_v2.png
|   |   |   |-- sagemaker_train_v2.ipynb    
|   |   |-- b/                            # Experiment 3b: depth=None (75.31%)
|   |   |   |-- sage_nopca_best.ipynb     # Code-only notebook
|   |   |   |-- sage_nopca_bestresults.ipynb # Notebook with output results
|   |   |   |-- leaf_disease_pipeline_best.pkl
|   |   |   |-- classification_report_best.txt
|   |   |   |-- confusion_matrix_best.png
|   |
|   |-- exp4/                             # Experiment 4: MobileNetV3-Small (98.13%)
|   |   |-- sage_mobilenet.ipynb          # Code-only starter notebook
|   |   |-- sage_mobilenetresult.ipynb    # Notebook with successful outputs (98.13%)
|   |   |-- mobilenet_leaf_disease.tflite # 1.1 MB TFLite model 
|   |   |-- mobilenet_leaf_disease.h5     # 8.7 MB Keras model
|   |   |-- classification_report_mobilenet.txt
|   |   |-- confusion_matrix_mobilenet.png
|   |   |-- training_curves_mobilenet.png
|   |
|   |-- work/                             # Documentation & Configuration
|       |-- T_WORK.md                 # Complete technical documentation of all 4 experiments
|       |-- comparison.md                 # T vs Vishnu work comparison
|       |-- continue.md                   # Future work, what to run next, post-testing updates
|       |-- METHODS_EXPLAINED.md          # Overview of old 7-feature vs new HOG+PCA methods
|       |-- Research_Paper_Draft.md       # Academic paper outline (5-class focus, early draft)
|       |-- STEP_BY_STEP_GUIDE.md         # Training and paper writing checklist
|       |-- kaggle.json                   # Kaggle API token (gitignored, not in repo)
```

---

## Platforms & Infrastructure

| Platform | Used By | Instance | Specs | Cost |
|----------|---------|----------|-------|------|
| Google Colab | Vishnu (all), T (Exp 1) | Free tier | CPU, limited RAM | Free |
| AWS SageMaker Studio | T (Exp 2, 3a, 3b) | ml.c5.4xlarge | 16 vCPU, 32 GB RAM, no GPU | On-demand |
| AWS SageMaker Studio | T (Exp 4) | ml.g4dn.xlarge | 4 vCPU, 16 GB RAM, 1x NVIDIA T4 GPU (16 GB VRAM) | $0.53/hr |

---

## Tools & Libraries

| Library | Version | Purpose | Used By |
|---------|---------|---------|---------|
| scikit-learn | 1.6.1 (Colab) | RandomForestClassifier, GradientBoostingClassifier, PCA, StandardScaler, train_test_split, accuracy_score, classification_report, confusion_matrix | Both |
| OpenCV (cv2) | 4.13.0.90 (Colab) | Image reading (imread), BGR->grayscale, Gaussian blur, resize | Both |
| NumPy | 2.0.2 (Colab) | Array operations, mean, std, min, max, median, var, hstack | Both |
| scikit-image | 0.25.2 (Colab) | HOG feature extraction (`skimage.feature.hog`) | T (Exp 2-3), Vishnu (in requirements but unused in training) |
| pandas | 2.2.2 (Colab) | Feature DataFrame, CSV read/write | Vishnu |
| matplotlib | 3.10.0 (Colab) | Confusion matrix plots, feature importance plots | T (Exp 1) |
| joblib | 1.5.3 (Colab) | Model serialization with compression (compress=3) | Both |
| Gradio | 6.5.1 | Web interface for inference | Vishnu (deployment) |
| PyYAML | — | Parsing YOLO dataset class names from classes.yaml | T |
| Kaggle API | — | Dataset download (`kaggle datasets download`) | Both |
| TensorFlow | 2.18.0 | MobileNetV3-Small transfer learning (Exp 4) | T |

---

## What Is Done

- [x] Vishnu: Downloaded PlantVillage 5-class subset from Kaggle (5,652 images, 658 MB)
- [x] Vishnu: Preprocessed images (128x128, Gaussian blur)
- [x] Vishnu: Extracted 7 color features (RGB mean/std + gray texture std) -> leaf_features.csv
- [x] Vishnu: Trained Gradient Boosting + PCA -> 62.86% accuracy (2 PCA components)
- [x] Vishnu: Trained Random Forest (500 trees, balanced) -> 83.90% accuracy (no PCA)
- [x] Vishnu: Saved model as leaf_disease_model.pkl
- [x] Vishnu: Created Gradio app.py with grayscale feature extraction
- [x] Vishnu: Deployed on Hugging Face Spaces (live at chinni-1221/crop-leaf-disease-detection)
- [x] T: Downloaded PlantVillage full 38-class YOLO dataset (54,293 images, ~829 MB)
- [x] T Exp 1: 7 grayscale features -> RF (depth=None) -> 44.48% accuracy, 699.8 MB model (Google Colab)
- [x] T Exp 1: Generated confusion_matrix.png, feature_importance.png, classification_report.txt
- [x] T Exp 2: 13 color + HOG (8x8) + PCA -> RF (depth=25) -> 53.55% accuracy (SageMaker ml.c5.4xlarge)
- [x] T Exp 2: Diagnosed PCA + RF incompatibility (root cause of low accuracy)
- [x] T Exp 3a: 13 color + HOG (16x16) no PCA -> RF (depth=25) -> 75.52% accuracy, 445.6 MB (SageMaker ml.c5.4xlarge)
- [x] T Exp 3b: Same as 3a with depth=None -> 75.31% accuracy, 511.1 MB (SageMaker ml.c5.4xlarge)
- [x] T: Proved ~75% is the hard feature ceiling (depth cap removal gave no improvement)
- [x] T: Analyzed and rejected Hybrid CNN+RF approach (two brains, size irony, wrong bottleneck)
- [x] T: Prepared MobileNetV3-Small notebook code (sage_mobilenetresult.ipynb), and successfully executed.
- [x] T Exp 4: Ran sage_mobilenetresult.ipynb on SageMaker ml.g4dn.xlarge (MobileNetV3-Small, transfer learning)
- [x] T Exp 4: Recorded actual accuracy (98.13%), .h5 size (8.7 MB), .tflite size (1.1 MB), training curves, and per-class reports
- [x] T: Created comprehensive documentation (T_WORK.md, comparison.md, continue.md, exp1_methodology.md)

## What Is Pending

- [ ] Paper: Write research paper using all experiment results, confusion matrices, feature importance charts, accuracy progression

---

## Known Issues

1. **Vishnu's training/inference feature mismatch:** The model was trained on 7 color features (RGB mean, RGB std, grayscale texture std) but the deployed `app.py` extracts 7 different grayscale statistical features (mean, std, min, max, median, variance, entropy). These are entirely different feature spaces — every position has a different meaning and range. The notebook also defines HOG extraction that is never called during training, and cell 27 writes a third version of app.py (with HOG) that doesn't match the deployed one.

2. **Persistent 0% recall classes (T):** Potato___healthy (152 total, 30 test samples) and Apple___Cedar_apple_rust (275 total, 55 test samples) scored 0% recall across ALL of T's experiments (Exp 1, 2, 3a, 3b). Extreme class imbalance is the primary cause — even unlimited tree depth and 1,777 features couldn't help.

3. **Large RF model files:** Random Forest models range from 445 MB to 700 MB depending on compression and depth settings. All .pkl files tracked with Git LFS. MobileNetV3 .tflite (Exp 4) is just 1.1 MB — a 400-600x size reduction.

4. **PCA + RF incompatibility:** Documented in Exp 2. Any future work using tree-based models (RF, XGBoost, LightGBM) should avoid PCA dimensionality reduction. Use larger pixels_per_cell in HOG instead to naturally reduce feature count.
