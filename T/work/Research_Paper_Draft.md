# Research Paper Draft: Crop Leaf Disease Detection

## Suggested Titles
1. **Lightweight Crop Leaf Disease Detection Using Statistical Feature Extraction and Random Forest Classification**
2. **Resource-Efficient Plant Disease Diagnosis: A 7-Feature Statistical Approach for Tomato and Potato Leaf Classification**
3. **Extreme Resource Efficiency in Precision Agriculture: Hand-Crafted Statistical Features for GPU-Free Leaf Disease Detection**
4. **Interpretable and Deployable Crop Disease Classification Using Statistical Image Analysis and Ensemble Learning**

---

## Abstract
Deep learning models for crop disease detection demand significant computational resources, making them impractical for deployment in low-resource agricultural settings. This study presents a lightweight, GPU-free methodology for classifying five crop health statuses (Potato Early Blight, Potato Healthy, Tomato Early Blight, Tomato Late Blight, and Tomato Healthy) using only seven hand-crafted statistical image features. Each input leaf image is converted to grayscale, resized to 128×128 pixels, and characterized by seven interpretable features: mean intensity, standard deviation, minimum pixel value, maximum pixel value, median, variance, and entropy. These features are classified using a Random Forest Classifier with 500 estimators and balanced class weights. The system achieves a baseline accuracy of 75–80% and is deployed as a real-time web application via Gradio on Hugging Face Spaces. Unlike CNN-based approaches, the proposed method requires no GPU for training or inference, is fully explainable, and demonstrates near-zero latency prediction — making it viable for edge deployment in resource-constrained environments.

---

## Paper Outline

### I. Introduction
- **Context:** Crop diseases (especially in Potato and Tomato) cause significant yield losses. Early detection is critical.
- **Problem:** Modern deep learning approaches (CNNs) require expensive GPUs for training and inference. Many agricultural regions lack access to such hardware.
- **Proposed Solution:** A classification system using 7 lightweight, interpretable statistical features extracted from leaf images, classified by a Random Forest ensemble model.
- **Contributions:**
  1. A minimal 7-feature extraction pipeline that captures disease-relevant image statistics.
  2. A Random Forest classifier that achieves 75–80% accuracy without any GPU.
  3. A real-time web deployment on Hugging Face Spaces via Gradio, proving practical usability.

### II. Related Work
- **Deep Learning approaches:** Discuss CNNs (ResNet, EfficientNet, VGG) applied to PlantVillage — high accuracy but heavy compute. Cite recent 2024–2025 papers.
- **Traditional ML approaches:** Reference papers using SVM, KNN, or Random Forest with texture features (GLCM, HOG, color histograms).
- **Gap:** Most recent papers use CNN-extracted features even for RF/SVM baselines. Very few use purely hand-crafted statistical features as a standalone system. This gap is your contribution.

### III. Dataset
- **Source:** PlantVillage dataset (publicly available).
- **Subset used:** Potato and Tomato leaves.
- **5 Classes:**
  | Class | Description |
  |---|---|
  | Potato_Early_blight | Diseased potato leaf |
  | Potato_healthy | Healthy potato leaf |
  | Tomato_Early_blight | Diseased tomato leaf (early stage) |
  | Tomato_Late_blight | Diseased tomato leaf (late stage) |
  | Tomato_healthy | Healthy tomato leaf |
- **Class imbalance:** Handled via `class_weight='balanced'` in the Random Forest.

### IV. Methodology

#### 4.1 Image Pre-processing
1. Convert input image to **grayscale** (removes color dependency, reduces dimensionality).
2. Resize to **128×128 pixels** (standardizes input, reduces computation).

#### 4.2 Feature Extraction (7 Features)
| # | Feature | Formula / Description | Why It Matters |
|---|---|---|---|
| 1 | Mean | Average pixel intensity | Overall brightness — diseased leaves may be darker |
| 2 | Standard Deviation | Spread of pixel values | Higher variation = more spots/lesions |
| 3 | Minimum | Darkest pixel value | Dark spots indicate necrotic tissue |
| 4 | Maximum | Brightest pixel value | Bright spots may indicate chlorosis |
| 5 | Median | Middle pixel value | Robust central tendency, less affected by outliers |
| 6 | Variance | Square of standard deviation | Amplifies differences between healthy and diseased |
| 7 | Entropy | −Σ(p × log₂p) from pixel histogram | Texture complexity — diseased leaves have higher entropy |

#### 4.3 Classification Model
- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Hyperparameters:**
  - `n_estimators=500` — 500 decision trees for robust ensemble predictions
  - `class_weight='balanced'` — automatically adjusts weights inversely proportional to class frequency

#### 4.4 Deployment
- **Framework:** Gradio
- **Hosting:** Hugging Face Spaces
- **Inference:** Real-time, CPU-only, near-zero latency

### V. Results & Discussion
- **Accuracy:** 75–80% across 5 classes.
- **Confusion Matrix:** *(generate using the provided script — see STEP_BY_STEP_GUIDE.md)*
- **Classification Report:** Precision, Recall, F1-score per class.
- **Inference Speed:** Predictions in <50ms on standard CPU (no GPU required).
- **Comparison with literature:** CNN-based methods achieve 90%+ but require GPU. This system trades ~15% accuracy for zero GPU dependency, full interpretability, and instant deployment.

### VI. Explainability & Practical Value
- **Interpretable features:** Unlike "black box" CNNs, each of the 7 features has a clear physical meaning.
- **Explainable AI (XAI):** A farmer or agronomist can understand *why* the model classified a leaf as diseased (e.g., "high variance and entropy indicate irregular spots").
- **Edge/IoT deployment:** The model is small enough to run on a Raspberry Pi or smartphone without internet.

### VII. Conclusion & Future Work
- **Summary:** A 7-feature statistical approach successfully provides GPU-free, interpretable, and deployable crop disease classification.
- **Limitations:** Accuracy is lower than CNN approaches; grayscale conversion discards color information that may be diagnostically useful.
- **Future Work:**
  - Add color-based features (mean R, G, B channels) for another 3 features.
  - Add texture features (GLCM: contrast, correlation, energy, homogeneity).
  - Expand to more crop classes (Apple, Corn, Grape).
  - Compare with lightweight CNNs (MobileNet) for a comprehensive benchmark.

---

## References (to find and cite)
1. PlantVillage dataset — Hughes & Salathé (2015)
2. Random Forest — Breiman (2001)
3. Recent CNN-based plant disease papers (2024–2025) — search Semantic Scholar
4. GLCM texture features — Haralick et al. (1973)
5. Gradio framework — Abid et al. (2019)
