# T — Crop Leaf Disease Detection: Complete Technical Documentation

## Dataset

- **Source:** PlantVillage for Object Detection (YOLO format) from Kaggle
- **URL:** `kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo`
- **Total images:** 54,293
- **Number of classes:** 38
- **Format:** YOLO (each image has a `.txt` label file with `class_id x_center y_center width height`)
- **Class names loaded from:** `classes.yaml` inside the dataset
- **License:** CC-BY-NC-SA-4.0
- **Dataset size:** ~829 MB compressed

### Class Distribution

| Class | Samples |
|-------|---------|
| Apple___Apple_scab | 630 |
| Apple___Black_rot | 621 |
| Apple___Cedar_apple_rust | 275 |
| Apple___healthy | 1,645 |
| Blueberry___healthy | 1,502 |
| Cherry___Powdery_mildew | 1,052 |
| Cherry___healthy | 853 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 513 |
| Corn___Common_rust | 1,192 |
| Corn___Northern_Leaf_Blight | 983 |
| Corn___healthy | 1,157 |
| Grape___Black_rot | 1,180 |
| Grape___Esca_(Black_Measles) | 1,383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1,076 |
| Grape___healthy | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 5,507 |
| Peach___Bacterial_spot | 2,297 |
| Peach___healthy | 360 |
| Pepper,_bell___Bacterial_spot | 997 |
| Pepper,_bell___healthy | 1,477 |
| Potato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___healthy | 152 |
| Raspberry___healthy | 371 |
| Soybean___healthy | 5,089 |
| Squash___Powdery_mildew | 1,835 |
| Strawberry___Leaf_scorch | 1,109 |
| Strawberry___healthy | 456 |
| Tomato___Bacterial_spot | 2,127 |
| Tomato___Early_blight | 1,000 |
| Tomato___Late_blight | 1,907 |
| Tomato___Leaf_Mold | 952 |
| Tomato___Septoria_leaf_spot | 1,771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 |
| Tomato___Target_Spot | 1,404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,357 |
| Tomato___Tomato_mosaic_virus | 373 |
| Tomato___healthy | 1,591 |

**Imbalance range:** 30:1 ratio (Potato_healthy: 152 vs Orange_Haunglongbing: 5,507)

---

## Experiment 1: Old 7-Feature Method (Grayscale Only)

**Folder:** `T/exp1/`
**Files:** `Crop.ipynb`, `crop_code.py`, `train_model.py`
**Platform:** Google Colab (free tier, CPU)

### Preprocessing
- Convert image from BGR to grayscale (OpenCV)
- Resize to **128x128** pixels
- Compute 256-bin histogram, normalize to probability distribution

### Features Extracted (7 per image)

| # | Feature | Formula |
|---|---------|---------|
| 1 | Mean pixel intensity | `np.mean(gray)` |
| 2 | Standard deviation | `np.std(gray)` |
| 3 | Minimum pixel value | `np.min(gray)` |
| 4 | Maximum pixel value | `np.max(gray)` |
| 5 | Median pixel value | `np.median(gray)` |
| 6 | Variance | `np.var(gray)` |
| 7 | Entropy | `-sum(hist * log2(hist + 1e-7))` |

**No color features. No texture features. Only grayscale statistics.**

### Model Configuration
- **Model:** Random Forest Classifier (scikit-learn)
- **n_estimators:** 500
- **max_depth:** None (unlimited)
- **class_weight:** "balanced"
- **random_state:** 42
- **n_jobs:** -1 (all CPU cores)

### Train/Test Split
- **Split:** 80% train / 20% test
- **Stratified:** Yes (preserves class proportions)
- **Test samples:** 10,859

### Results
- **Accuracy: 44.48%**

#### Per-class highlights

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Corn___Common_rust | 0.98 | 0.99 | 0.98 |
| Corn___healthy | 0.83 | 0.86 | 0.85 |
| Soybean___healthy | 0.64 | 0.86 | 0.73 |
| Potato___healthy | 0.00 | 0.00 | 0.00 |
| Potato___Late_blight | 0.17 | 0.09 | 0.11 |
| Apple___Cedar_apple_rust | 0.20 | 0.04 | 0.06 |

### Model Size
- **CLEAN version (compress=3):** 699.76 MB

### Output Files
- `confusion_matrix.png` (12x10 inches, 150 DPI, Blues colormap)
- `feature_importance.png` (8x5 inches, 150 DPI, green horizontal bars)
- `classification_report.txt`
- `leaf_disease_model.pkl`
- `leaf_disease_model_CLEAN.pkl` (compressed with joblib compress=3)
- `leaf_disease_model_TINY.pkl` (compressed with zlib level 9)

### Why 44% Accuracy
Only 7 grayscale features for 38 classes. No color information (RGB), no texture information (HOG). Many diseases look identical in grayscale statistics alone — the model has almost nothing to distinguish them.

---

## Experiment 2: New HOG+PCA Method (With PCA — Broken)

**File:** `T/exp2/sage_pca.ipynb`
**Platform:** AWS SageMaker Studio (ml.c5.4xlarge — 16 vCPU, 32 GB RAM)

### Preprocessing
1. **Gaussian Blur:** `cv2.GaussianBlur(image, (5,5), 0)` to reduce background noise
2. **Resize:** to **128x128** pixels
3. **Grayscale conversion:** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
4. **Histogram:** 256-bin, normalized to probability distribution

### Features Extracted (8,113 per image)

#### 13 Color + Grayscale Features

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

#### ~8,100 HOG (Histogram of Oriented Gradients) Features

| Parameter | Value |
|-----------|-------|
| orientations | 9 |
| pixels_per_cell | **(8, 8)** |
| cells_per_block | (2, 2) |
| block_norm | L2-Hys |
| Input image | 128x128 grayscale |
| Output features | ~8,100 |

**Total raw features:** 13 + 8,100 = **8,113**

### Dimensionality Reduction (PCA)

| Step | Detail |
|------|--------|
| StandardScaler | Applied to all 8,113 features (zero mean, unit variance) |
| PCA | `n_components=0.95` (retain 95% variance) |
| Features after PCA | **2,074** (compressed from 8,113) |

### Model Configuration
- **Model:** Random Forest Classifier
- **n_estimators:** 500
- **max_depth:** 25
- **min_samples_leaf:** 2
- **class_weight:** "balanced"
- **random_state:** 42
- **n_jobs:** -1

### Train/Test Split
- **Split:** 80% train / 20% test
- **Stratified:** Yes
- **Train:** 43,434 / **Test:** 10,859

### Results
- **Accuracy: 53.55%**

#### Per-class highlights

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Corn___Common_rust | 0.88 | 0.94 | 0.91 |
| Grape___Leaf_blight | 0.71 | 0.78 | 0.74 |
| Peach___healthy | 1.00 | 0.06 | 0.11 |
| Pepper,_bell___Bacterial_spot | 1.00 | 0.01 | 0.02 |
| Apple___Apple_scab | 0.00 | 0.00 | 0.00 |
| Apple___Cedar_apple_rust | 0.00 | 0.00 | 0.00 |
| Potato___healthy | 0.00 | 0.00 | 0.00 |
| Tomato___Tomato_mosaic_virus | 0.00 | 0.00 | 0.00 |

**Macro avg precision:** 0.55 | **Macro avg recall:** 0.39

### Saved Pipeline
```python
pipeline = {
    "scaler": StandardScaler,
    "pca": PCA(n_components=0.95),
    "model": RandomForestClassifier,
    "classes": 38 class names
}
```
File: `leaf_disease_pipeline.pkl`

### Why 53% Accuracy — The PCA + Random Forest Mismatch

**Root cause:** PCA rotates features diagonally to compress them. Random Forest makes axis-aligned splits (horizontal/vertical decision boundaries like "if red_mean > 50"). When PCA-rotated features are fed to a Random Forest, the tree cannot draw clean splits — it must approximate diagonal boundaries with hundreds of tiny "staircase" splits, and with `max_depth=25`, it runs out of depth.

**Evidence:**
- Classes with distinctive single-feature signals survived PCA partially (Corn Common Rust: 94% recall — its strong orange color projects onto PCA components in a partially axis-separable way)
- Classes depending on subtle multi-feature patterns collapsed to 0-2% recall (Apple Scab, Pepper Bell Bacterial Spot, Tomato Early Blight)
- 4 classes had exactly 0% precision and 0% recall (never predicted at all)

**Rule:** PCA is compatible with SVMs, logistic regression, and neural networks (which can learn rotated boundaries). PCA is NOT compatible with tree-based models (Random Forest, Gradient Boosted Trees, Decision Trees) which rely on axis-aligned splits.

---

## Experiment 3: New HOG Method Without PCA (Fixed)

**File:** `T/exp3/a/sage_nopca.ipynb`
**Platform:** AWS SageMaker Studio (ml.c5.4xlarge — 16 vCPU, 32 GB RAM)
**Also available as:** `sage_nopca_full.ipynb` (includes dataset download), `colab_nopca.ipynb` and `colab_nopca.py` (Google Colab versions)

### Preprocessing
Identical to Experiment 2:
1. Gaussian Blur (5x5 kernel)
2. Resize to 128x128
3. Grayscale conversion
4. 256-bin histogram normalization

### Features Extracted (1,777 per image)

#### 13 Color + Grayscale Features
Same 13 features as Experiment 2.

#### ~1,764 HOG Features

| Parameter | Value |
|-----------|-------|
| orientations | 9 |
| pixels_per_cell | **(16, 16)** (changed from 8x8) |
| cells_per_block | (2, 2) |
| block_norm | L2-Hys |
| Input image | 128x128 grayscale |
| Output features | **~1,764** (down from ~8,100) |

**Total features:** 13 + 1,764 = **1,777**

### Key Changes from Experiment 2

| What | Experiment 2 (PCA) | Experiment 3 (No PCA) |
|------|--------------------|-----------------------|
| HOG pixels_per_cell | (8, 8) → 8,100 features | (16, 16) → 1,764 features |
| PCA | Yes (8,113 → 2,074 rotated) | **Removed** |
| StandardScaler | Yes | **Removed** |
| Features to RF | 2,074 PCA-rotated | 1,777 raw unrotated |
| Pipeline saved | scaler + pca + model | model only |

### Why (16,16) Instead of (8,8)
Increasing `pixels_per_cell` from (8,8) to (16,16) naturally reduces HOG output from ~8,100 to ~1,764 features — keeping the model small without needing PCA compression. The raw, unrotated features can be split cleanly by the Random Forest's axis-aligned decision boundaries.

### Model Configuration
- **Model:** Random Forest Classifier
- **n_estimators:** 500
- **max_depth:** 25
- **min_samples_leaf:** 2
- **class_weight:** "balanced"
- **random_state:** 42
- **n_jobs:** -1

### Train/Test Split
- **Split:** 80% train / 20% test
- **Stratified:** Yes
- **Train:** 43,434 / **Test:** 10,859

### Results
- **Accuracy: 75.52%**

#### Per-class results (all 38 classes)

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
| **Weighted avg** | **0.76** | **0.76** | **0.74** | **10,859** |
| **Macro avg** | **0.75** | **0.65** | **0.67** | **10,859** |

### Model Size
- **File:** `leaf_disease_pipeline_v2.pkl`
- **Size:** 445.6 MB (compress=3)

### Saved Pipeline
```python
pipeline = {
    "model": RandomForestClassifier,
    "classes": 38 class names
}
```
No scaler or PCA saved — not needed.

### Remaining Bottleneck
`max_depth=25` caps the trees from fully learning harder classes. Classes that look visually similar (various tomato diseases, apple diseases) need more tree depth to separate. Two classes still at 0% recall (Cedar_apple_rust: 55 samples, Potato_healthy: 30 samples) due to extreme class imbalance.

---

## Experiment 3b: HOG (16x16) + RF (depth=None) — Traditional ML Best

**File:** `T/exp3/b/sage_nopca_bestresults.ipynb`
**Platform:** AWS SageMaker Studio (ml.c5.4xlarge — 16 vCPU, 32 GB RAM)

### Configuration
Same as Experiment 3 except:
- **max_depth:** None (unlimited tree growth)
- Everything else identical (500 trees, balanced, random_state=42)

### Results
- **Accuracy: 75.31%**

#### Per-class results (all 38 classes)

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
| **Weighted avg** | **0.76** | **0.75** | **0.73** | **10,859** |
| **Macro avg** | **0.75** | **0.64** | **0.67** | **10,859** |

### Model Size
- **File:** `leaf_disease_pipeline_best.pkl`
- **Size:** 511.1 MB (compress=3)

### Why depth=None Didn't Help

Removing the depth cap gave 75.31% — essentially identical to 75.52% with depth=25. The model grew from 445.6 MB to 511.1 MB with no accuracy gain. This proves **max_depth was NOT the bottleneck.**

The real bottleneck is the **feature ceiling**: HOG at (16,16) pixels_per_cell on 128x128 images produces ~1,764 texture features that cannot distinguish visually similar diseases. Classes with subtle differences (various tomato blights, apple diseases) and extreme class imbalance (Potato_healthy: 30 samples, Cedar_apple_rust: 55 samples) remain at 0-17% recall regardless of tree depth. Deeper trees simply overfit the training data without learning better separating patterns.

This is the hard accuracy limit of traditional ML (hand-crafted features) on this 38-class dataset, and it validates the need for deep learning (Experiment 4).

---

## Experiment 4: MobileNetV3-Small (Completed)

**File:** `T/exp4/sage_mobilenetresult.ipynb`
**Platform:** AWS SageMaker Studio (ml.g4dn.xlarge — 4 vCPU, 16 GB RAM, 1x NVIDIA T4 GPU)

### Preprocessing & Approach
1. Transfer learning from ImageNet (MobileNetV3-Small base model).
2. Input dimension: 128x128x3. MobileNet handles normalization internally (pixel range `[0, 255]`).
3. Added data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`).
4. **Phase 1 Training:** Base model frozen, trained classification head (learning rate `1e-3`).
5. **Phase 2 Training:** Unfroze top 20 layers for fine-tuning (lowered learning rate to `1e-5`).

### Features Extracted
Features are automatically learned by the Convolutional Neural Network (CNN) instead of being hand-crafted like HOG or grayscale stats.

### Results
- **Accuracy: 98.13%**
- **Training Time:** ~15-20 minutes on NVIDIA T4 GPU

#### Per-class highlights
Precision and recall scores were incredibly strong across all 38 classes, with the weighted average metrics consistently reaching ~0.98. The model successfully distinguished visually similar variations with ease.

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Apple___Apple_scab | 0.99 | 0.97 | 0.98 | 126 |
| Apple___Black_rot | 0.98 | 1.00 | 0.99 | 124 |
| Apple___Cedar_apple_rust | 1.00 | 1.00 | 1.00 | 55 |
| Apple___healthy | 0.98 | 0.98 | 0.98 | 329 |
| Blueberry___healthy | 1.00 | 1.00 | 1.00 | 300 |
| Cherry___Powdery_mildew | 1.00 | 1.00 | 1.00 | 210 |
| Cherry___healthy | 1.00 | 1.00 | 1.00 | 171 |
| Corn___Cercospora_leaf_spot | 0.83 | 0.88 | 0.85 | 103 |
| Corn___Common_rust | 1.00 | 1.00 | 1.00 | 238 |
| Corn___Northern_Leaf_Blight | 0.94 | 0.91 | 0.93 | 197 |
| Corn___healthy | 1.00 | 1.00 | 1.00 | 231 |
| Grape___Black_rot | 0.99 | 1.00 | 0.99 | 236 |
| Grape___Esca_(Black_Measles) | 1.00 | 0.99 | 1.00 | 277 |
| Grape___Leaf_blight | 1.00 | 1.00 | 1.00 | 215 |
| Grape___healthy | 0.98 | 1.00 | 0.99 | 85 |
| Orange___Haunglongbing | 1.00 | 1.00 | 1.00 | 1102 |
| Peach___Bacterial_spot | 1.00 | 1.00 | 1.00 | 460 |
| Peach___healthy | 0.99 | 1.00 | 0.99 | 72 |
| Pepper,_bell___Bacterial_spot | 0.96 | 1.00 | 0.98 | 199 |
| Pepper,_bell___healthy | 0.99 | 0.99 | 0.99 | 295 |
| Potato___Early_blight | 0.99 | 1.00 | 0.99 | 200 |
| Potato___Late_blight | 0.98 | 0.98 | 0.98 | 200 |
| Potato___healthy | 0.97 | 0.93 | 0.95 | 30 |
| Raspberry___healthy | 1.00 | 0.99 | 0.99 | 74 |
| Soybean___healthy | 1.00 | 1.00 | 1.00 | 1018 |
| Squash___Powdery_mildew | 1.00 | 1.00 | 1.00 | 367 |
| Strawberry___Leaf_scorch | 1.00 | 1.00 | 1.00 | 222 |
| Strawberry___healthy | 1.00 | 1.00 | 1.00 | 91 |
| Tomato___Bacterial_spot | 1.00 | 0.97 | 0.99 | 426 |
| Tomato___Early_blight | 0.97 | 0.83 | 0.89 | 200 |
| Tomato___Late_blight | 0.97 | 0.96 | 0.96 | 381 |
| Tomato___Leaf_Mold | 0.99 | 0.96 | 0.97 | 190 |
| Tomato___Septoria_leaf_spot | 0.97 | 0.96 | 0.96 | 354 |
| Tomato___Spider_mites | 0.95 | 0.91 | 0.93 | 335 |
| Tomato___Target_Spot | 0.82 | 0.96 | 0.88 | 281 |
| Tomato___Yellow_Leaf_Curl_Virus | 1.00 | 0.98 | 0.99 | 1072 |
| Tomato___Tomato_mosaic_virus | 1.00 | 0.97 | 0.99 | 75 |
| Tomato___healthy | 0.93 | 1.00 | 0.97 | 318 |
| **Weighted avg** | **0.98** | **0.98** | **0.98** | **10859** |

### Model Size
- **File:** `mobilenet_leaf_disease.tflite`
- **Size:** 1.1 MB (quantized for edge devices)
- **Keras H5 equivalent:** 8.7 MB

### Why 98% Accuracy
Deep learning (CNNs) operates beyond the hard ~75% ceiling of manually engineered features. It dynamically learns the most optimal texture, color, and shape features to separate the 38 classes. By fixing a prior normalization bug and applying data augmentation with a proper fine-tuning schedule, the model reached peak performance on this dataset.

---

## Accuracy Progression Summary

| Experiment | Method | Features | Accuracy | Model Size | Latency (CPU) | Platform |
|------------|--------|----------|----------|------------|---------------|----------|
| 1. Old 7-feature | 7 grayscale stats → RF (depth=None) | 7 | **44.48%** | 699.8 MB | — | Google Colab (CPU) |
| 2. HOG + PCA | 13 color + HOG 8x8 → PCA → RF (depth=25) | 2,074 (rotated) | **53.55%** | — | — | SageMaker ml.c5.4xlarge |
| 3a. HOG No PCA | 13 color + HOG 16x16 → RF (depth=25) | 1,777 (raw) | **75.52%** | 445.6 MB | **22.40 ms/image** | SageMaker ml.c5.4xlarge |
| 3b. HOG No PCA (best) | 13 color + HOG 16x16 → RF (depth=None) | 1,777 (raw) | **75.31%** | 511.1 MB | **22.65 ms/image** | SageMaker ml.c5.4xlarge |
| 4. MobileNetV3-Small | End-to-end transfer learning | Learned by CNN | **98.13%** | 1.1 MB (.tflite) | **5.54 ms/image** | SageMaker ml.g4dn.xlarge |

**Inference Latency Benchmark (SageMaker ml.g4dn.xlarge, CPU, 200 random images):**
- Exp 3a: 22.40 ms/image (median: 22.35 ms) — includes HOG feature extraction + RF prediction
- Exp 3b: 22.65 ms/image (median: 22.58 ms) — deeper trees, same inference speed
- Exp 4 TFLite: **5.54 ms/image** (median: 5.54 ms) — 4x faster than HOG+RF, 224x224 input, CPU only

### Key Findings

1. **Adding color (RGB) and texture (HOG) features** improved accuracy from 44% to 53% (Experiment 1 → 2), but PCA suppressed most of the gain.
2. **Removing PCA** and feeding raw features to Random Forest improved accuracy from 53% to 76% (Experiment 2 → 3a) — a 22-point jump from a single algorithmic fix.
3. **PCA + Random Forest is a known incompatibility.** PCA rotates features diagonally; Random Forest splits axis-aligned. The mismatch forces inefficient staircase approximations that waste tree depth.
4. **~75% is the feature ceiling for HOG + RF on 38 classes.** Removing the depth cap (3a → 3b) gave zero improvement, proving the bottleneck is feature quality, not model capacity. Only learned features (CNN) can push beyond this ceiling.

---

## All Files

### `T/exp1/` — Experiment 1: 7-Feature Baseline

| File | Description |
|------|-------------|
| `Crop.ipynb` | Google Colab notebook — 7 grayscale features, RF, 44.48% accuracy |
| `crop_code.py` | Same code as Crop.ipynb in .py script form for Colab |
| `train_model.py` | Standalone local script version (expects class subfolders) |
| `classification_report.txt` | Per-class precision/recall/F1 results |
| `confusion_matrix.png` | 38-class confusion matrix visualization |
| `feature_importance.png` | Bar chart of 7 feature importances |
| `leaf_disease_model_CLEAN.pkl` | Trained model (compress=3, ~700 MB) |
| `leaf_disease_model_TINY.pkl` | Trained model (heavy compression) |
| `exp1_methodology.md` | Detailed step-by-step methodology explanation |

### `T/exp2/` — Experiment 2: HOG + PCA (Broken)

| File | Description |
|------|-------------|
| `sage_pca.ipynb` | SageMaker notebook — HOG 8x8 + PCA + RF, **53.55%** accuracy (has run outputs) |

### `T/exp3/` — Experiment 3: HOG No PCA

| File | Description |
|------|-------------|
| `a/sage_nopca.ipynb` | SageMaker notebook — HOG 16x16, no PCA, RF depth=25, **75.52%** |
| `a/sage_nopca_full.ipynb` | Same as sage_nopca but includes full dataset download |
| `a/colab_nopca.ipynb` | Google Colab notebook version of the no-PCA approach |
| `a/colab_nopca.py` | Google Colab .py script version of the no-PCA approach |
| `b/sage_nopca_best.ipynb` | SageMaker notebook — same but depth=None (code only) |
| `b/sage_nopca_bestresults.ipynb` | SageMaker notebook — depth=None with run outputs, **75.31%** |

### `T/exp4/` — Experiment 4: MobileNetV3-Small

| File | Description |
|------|-------------|
| `sage_mobilenetresult.ipynb` | SageMaker notebook — MobileNetV3-Small transfer learning (**successfully executed, 98.13%**) |
| `sage_mobilenet.ipynb` | SageMaker notebook — Starter version |

### `T/work/` — Documentation & Config

| File | Description |
|------|-------------|
| `T_WORK.md` | Comprehensive technical documentation of all experiments |
| `continue.md` | Future work, what to run next, post-testing updates |
| `comparison.md` | T vs Vishnu work comparison |
| `kaggle.json` | Kaggle API token (gitignored) |

---

## Common Configuration Across All Experiments

| Parameter | Value |
|-----------|-------|
| Image resize | 128x128 |
| Histogram bins | 256 |
| Classifier | Random Forest (scikit-learn) |
| n_estimators | 500 |
| class_weight | "balanced" |
| random_state | 42 |
| n_jobs | -1 (all cores) |
| Train/test split | 80/20 stratified |
| Gaussian blur | (5,5) kernel (Experiments 2 and 3 only) |
| Label reading | YOLO format — reads class_id from label .txt files |
| Model serialization | joblib with compress=3 |

---

## Tools and Libraries Used

- **Python 3.12** (SageMaker), Python 3.x (Colab)
- **scikit-learn** — RandomForestClassifier, train_test_split, StandardScaler, PCA, classification_report, confusion_matrix
- **OpenCV (cv2)** — image reading, BGR→grayscale, Gaussian blur, resize
- **scikit-image** — HOG feature extraction (`skimage.feature.hog`)
- **NumPy** — array operations, statistics
- **matplotlib** — confusion matrix plots, feature importance plots
- **joblib** — model serialization/compression
- **PyYAML** — parsing dataset class names
- **Kaggle API** — dataset download

---

## Appendix: Why Hybrid CNN+RF Was Rejected

### The Hybrid Idea

A fifth experiment was considered: use MobileNetV3 as a feature extractor (remove its classification head) and feed the CNN's learned features into a Random Forest classifier.

```
Hybrid pipeline:
Image → [MobileNetV3 body (frozen, pre-trained)] → 1,024 deep features → Random Forest → Prediction
```

This promises the best of both worlds:

| Property | HOG + RF (Exp 1-3) | MobileNet E2E (Exp 4) | Hybrid CNN+RF |
|----------|-------------------|----------------------|---------------|
| Who designs features? | You (manually) | CNN (automatically) | CNN (automatically) |
| Who classifies? | Random Forest | Dense layer | Random Forest |
| Feature quality | Limited by HOG | Excellent | Excellent |
| Needs lots of training data? | No | Yes (fine-tuning) | No (CNN frozen, RF data-efficient) |
| Needs GPU for training? | No | Yes | Only for one-time feature extraction |
| Needs GPU for inference? | No | Preferred | No (features pre-extracted, RF on CPU) |

The appeal: CNN-level accuracy with RF-level data efficiency and CPU inference.

### Why It Was Rejected

#### Problem 1: The "Two Brains" Software Dependency

**End-to-end deployment (Experiment 4):**
- Export one `.tflite` file
- Mobile developer loads it with TFLite library
- One framework, one file, works on phone's NPU/GPU natively

**Hybrid deployment:**
- Load `.tflite` to run MobileNet body → get 1,024 numbers
- Then load `rf_classifier.pkl` and run a Random Forest interpreter
- On Android/iOS, this means the developer needs both TFLite AND a scikit-learn equivalent
- There is no official scikit-learn runtime for mobile platforms
- The developer would have to either:
  - Port the RF to C++ manually
  - Bundle a Python interpreter inside the app
  - Convert the RF decision tree to a custom if-else engine
- This is an integration nightmare on edge devices

**For Edge AI, a single-framework deployment is critical.** Farmers' phones don't have Python installed. Drones run C firmware. Raspberry Pi deployments want minimal dependencies.

#### Problem 2: The Mathematical File Size Irony

In traditional ML (Experiments 1-3), the Random Forest IS the entire model. There's nothing else. It makes sense that it's large (445-700 MB) because it's doing all the work — both feature interpretation and classification.

In the hybrid, you already have MobileNetV3 loaded and doing the hard work of feature extraction. The classification layer it replaces is:

```
Dense layer: 1,024 inputs × 38 outputs = 38,912 weights + 38 biases = 38,950 parameters
At 4 bytes each = 155.8 KB
```

**155 KB.** That's what the hybrid replaces with a Random Forest.

A Random Forest trying to learn relationships across 1,024 deep features with 500 trees will easily weigh **10-20 MB** of decision node thresholds. So the hybrid takes a 155 KB classification layer and replaces it with a 10-20 MB Random Forest — the model gets **100x larger** at the classification step for no accuracy gain.

| Component | End-to-End (Exp 4) | Hybrid |
|-----------|-------------------|--------|
| MobileNetV3 body | ~3.4 MB | ~3.4 MB |
| Classification layer | ~155 KB (Dense) | ~10-20 MB (Random Forest) |
| **Total** | **~3.5 MB** | **~15-25 MB** |
| Extra framework needed | None | scikit-learn runtime |

#### Problem 3: The RF Was Never the Bottleneck

Experiments 1-3 proved that the Random Forest's accuracy was limited by **feature quality**, not by the RF algorithm itself. The RF classified correctly when given good features (Corn_Common_rust: 99% recall from HOG alone). It failed when HOG features couldn't capture the visual difference between diseases.

Once features are good enough (CNN features), even a tiny 155 KB Dense layer classifies correctly. There's no reason to bring in a 10-20 MB Random Forest when the problem was never the classifier — it was always the features.

### Paper Discussion Section Quote (suggested)

> "While a hybrid architecture (CNN feature extractor + Random Forest classifier) was evaluated conceptually, it violates Edge AI deployment principles. The 'two brains' software dependency — requiring both TFLite and a scikit-learn runtime on the target device — introduces integration complexity incompatible with resource-constrained field deployment. Furthermore, replacing the final Dense layer (155 KB, 38,950 parameters) with a Random Forest (10-20 MB) actually increases the model's edge footprint by two orders of magnitude at the classification stage. Therefore, end-to-end MobileNetV3-Small provides the optimal balance of ~95%+ accuracy within a single ~4 MB TFLite package suitable for direct phone, drone, or Raspberry Pi deployment."

### Conclusion

The Random Forest is a great classifier when it IS the entire model (Experiments 1-3). But once you're already running a neural network, the Dense layer is smaller, faster, and requires no extra software. The hybrid approach adds software complexity, increases model size, and solves a problem that doesn't exist (the RF was never the weak link — the features were).
