# Step-by-Step Guide: Train Random Forest Model & Write Paper

---

## About the Dataset & GitHub
- **Do NOT upload the dataset images to GitHub.** They are too large.
- Your GitHub repo should contain: `app.py`, `requirements.txt`, `README.md`, `train_model.py`, and `leaf_disease_model.pkl` (tracked via Git LFS).

---

## Part 1: Prepare the Dataset for Random Forest

The PlantVillage dataset on Kaggle has image folders organized by class. Random Forest does not read images directly — you must extract features first. Here is how:

### Step 1 — Download the PlantVillage Dataset
Download from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
Or use the subset with these 5 folders:
```
PlantVillage/
├── Potato___Early_blight/      (images)
├── Potato___healthy/           (images)
├── Tomato___Early_blight/      (images)
├── Tomato___Late_blight/       (images)
└── Tomato___healthy/           (images)
```

### Step 2 — Run `train_model.py` to Extract Features & Train

Use the `train_model.py` script (included in this repo). It does the following automatically:
1. Reads every image from all 5 class folders.
2. Converts each image to grayscale + resizes to 128×128.
3. Extracts 7 statistical features per image.
4. Trains a Random Forest Classifier (500 trees, balanced class weights).
5. Saves the model as `leaf_disease_model.pkl`.
6. Prints accuracy, confusion matrix, and classification report.

**Run it:**
```bash
python train_model.py --dataset_path ./PlantVillage
```

### Step 3 — Collect Results for Paper
After training, the script outputs:
- **Accuracy** (e.g., 78%)
- **Confusion Matrix** (saved as `confusion_matrix.png`)
- **Classification Report** (precision, recall, F1 per class — saved as `classification_report.txt`)
- **Feature Importance Chart** (saved as `feature_importance.png`)

These files go directly into your paper.

---

## Part 2: Understanding the Feature Extraction

Your `app.py` extracts these 7 features from each leaf image:

```
Image → Grayscale → Resize 128×128 → Extract 7 numbers → Random Forest → Disease label
```

| # | Feature | What it captures |
|---|---|---|
| 1 | Mean | Average pixel brightness |
| 2 | Standard Deviation | Spread of pixel values (spots = high std) |
| 3 | Min pixel | Darkest pixel (necrotic/dead tissue) |
| 4 | Max pixel | Brightest pixel (chlorosis/yellowing) |
| 5 | Median | Robust center value |
| 6 | Variance | Like std but amplified — separates healthy vs diseased |
| 7 | Entropy | Texture complexity (diseased = more complex) |

---

## Part 3: Write the Paper

### Structure
| Section | Content | Length |
|---|---|---|
| Abstract | Problem + 7 features + RF + accuracy + deployment | ~200 words |
| 1. Introduction | Why crop disease detection matters, why lightweight | 1–1.5 pages |
| 2. Related Work | CNN approaches (cite them), gap in lightweight methods | 1 page |
| 3. Dataset | PlantVillage, 5 classes, class imbalance | 0.5 page |
| 4. Methodology | Feature extraction diagram + RF hyperparameters | 1.5 pages |
| 5. Results | Accuracy, Confusion Matrix, Classification Report | 1.5 pages |
| 6. Discussion | Why this works without GPU, explainability advantage | 0.5 page |
| 7. Conclusion | Summary + future work (add GLCM, color features) | 0.5 page |

### Figures to Include
1. `confusion_matrix.png` — from training script
2. `feature_importance.png` — from training script
3. Screenshot of Gradio app running
4. Architecture diagram (Image → Preprocessing → 7 Features → Random Forest → Label)

---

## Summary Checklist
```
✅ 1. Download PlantVillage dataset (5 class folders)
✅ 2. Run train_model.py → generates model + charts
✅ 3. Copy accuracy numbers + charts into paper
✅ 4. Write paper using Research_Paper_Draft.md outline
✅ 5. Push code to GitHub (no dataset images!)
```
