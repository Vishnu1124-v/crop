# Comparison: T's Work vs Vishnu's Work

## Overview

Both team members worked on crop leaf disease detection using the PlantVillage dataset, but with fundamentally different scopes and approaches.

| Aspect | Vishnu | T |
|--------|--------|-------|
| **Dataset** | PlantVillage (standard folder format) | PlantVillage YOLO format |
| **Classes** | 5 (Potato + Tomato only) | 38 (all plant diseases) |
| **Images** | ~5,000 (5-class subset) | 54,293 (full dataset) |
| **Platform** | Google Colab (free tier) | Google Colab (Exp 1), SageMaker ml.c5.4xlarge (Exp 2-3), SageMaker ml.g4dn.xlarge (Exp 4) |
| **Deployment** | Gradio on Hugging Face Spaces | Edge AI (paper focus) |
| **Experiments** | 1 (single approach) | 4 (progressive experimentation) |

---

## Vishnu's Work

**Folder:** `vishnu/`
**Files:** `project_code.py`, `Project.ipynb`, `app.py`, `leaf_disease_model.pkl`, `README.md`

### Approach
- **5 classes:** Tomato_healthy, Tomato_Early_blight, Tomato_Late_blight, Potato___healthy, Potato___Early_blight
- **Features:** 7 features — RGB mean (3), RGB std (3), grayscale texture std (1)
- **Pipeline:** StandardScaler → PCA (95% variance) → Gradient Boosting (tried), then Random Forest (500 trees, balanced)
- **Deployment:** Gradio web app on Hugging Face Spaces

### Code Structure
1. Dataset download from Kaggle (`emmarex/plantdisease`)
2. Extract 5-class subset into `dataset/` folder
3. Feature extraction → save to `leaf_features.csv`
4. Train Gradient Boosting + Random Forest
5. Save model as `leaf_disease_model.pkl`
6. Gradio app for inference (`app.py`)

### Key Observation
The deployed `app.py` uses a **different feature set** (7 grayscale stats: mean, std, min, max, median, variance, entropy) than the training code (7 color features: RGB mean, RGB std, gray texture). The `project_code.py` also defines a HOG extraction function that is never called during training. This means the deployed model may not work correctly due to the training/inference feature mismatch.

### Result
- Working Gradio app deployed on Hugging Face Spaces
- Model trained on 5 classes with ~7 color features
- Accuracy: **83.90%** (RF), **62.86%** (Gradient Boosting + PCA)

---

## T's Work

**Folder:** `T/` (organized into `exp1/`, `exp2/`, `exp3/a/`, `exp3/b/`, `exp4/`, `work/`)

### Approach
Progressive experimentation on all 38 classes, building toward an Edge AI research paper.

### Experiments

| Exp | Method | Features | Accuracy | Model Size | Latency (CPU) |
|-----|--------|----------|----------|------------|---------------|
| 1 | 7 grayscale stats → RF (depth=None) | 7 | 44.48% | 699.8 MB | — |
| 2 | 13 color + HOG 8x8 → PCA → RF (depth=25) | 2,074 (rotated) | 53.55% | — | — |
| 3a | 13 color + HOG 16x16 → RF (depth=25) | 1,777 (raw) | 75.52% | 445.6 MB | 22.40 ms/image |
| 3b | 13 color + HOG 16x16 → RF (depth=None) | 1,777 (raw) | 75.31% | 511.1 MB | 22.65 ms/image |
| 4 | MobileNetV3-Small (transfer learning) | Learned by CNN | 98.13% | 1.1 MB (.tflite) | 5.54 ms/image |

### Key Findings
1. PCA + Random Forest is incompatible (axis-aligned splits vs rotated features)
2. ~75% is the hard ceiling for hand-crafted features on 38 classes
3. Deep learning (Experiment 4) broke through this ceiling, achieving 98.13% accuracy with a 1.1 MB model

---

## Direct Comparison

### Dataset Scope

| | Vishnu | T |
|---|---|---|
| Dataset source | `emmarex/plantdisease` (folder format) | `sebastianpalaciob/plantvillage-for-object-detection-yolo` |
| Total classes | 5 | 38 |
| Total images | ~5,000 | 54,293 |
| Class imbalance | Mild (5 similar-sized classes) | Severe (30:1 ratio) |

T's 38-class problem is fundamentally harder — diseases across different plant species look similar, and class imbalance is extreme.

### Feature Engineering

| | Vishnu | T |
|---|---|---|
| Color features | RGB mean + RGB std (6) | RGB mean + RGB std (6) |
| Grayscale features | Texture std (1) | Mean, std, min, max, median, variance, entropy (7) |
| HOG features | Defined but unused in training | Used in Exp 2-3 (1,764 features) |
| Total features | 7 | 7 (Exp 1) → 1,777 (Exp 3) |
| PCA | Yes (on 7 features) | Tried and removed (Exp 2 → 3) |

### Model & Deployment

| | Vishnu | T |
|---|---|---|
| Classifier | Gradient Boosting + RF | RF (Exp 1-3), MobileNetV3 (Exp 4) |
| Deployment target | Web app (Gradio/HuggingFace) | Edge devices (phone, drone, RPi) |
| Model format | `.pkl` (scikit-learn) | `.pkl` (Exp 1-3), `.tflite` (Exp 4) |
| Production-ready? | Yes (live on HuggingFace) | Experiments completed (paper phase) |

### Contributions

**Vishnu:** Built a working end-to-end prototype — data download, feature extraction, training, and live deployment on Hugging Face Spaces. Focused on delivering a usable product for 5 common diseases.

**T:** Deep technical investigation — diagnosed PCA+RF incompatibility, proved the 75% feature ceiling through ablation experiments, designed progressive experiment arc for a research paper. Focused on understanding why traditional ML fails and demonstrating the need for deep learning on edge devices.

---

## How They Complement Each Other

Vishnu's work provides the **deployment pipeline** (Gradio app, HuggingFace hosting), while T's work provides the **research depth** (why certain approaches fail, what the accuracy limits are, and how deep learning overcomes them). Together they cover both the practical engineering and the scientific analysis needed for a complete project.
