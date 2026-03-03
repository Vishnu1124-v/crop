# 🌿 Crop Leaf Disease Detection — Technical Methodology

> **16,384 pixels → 7 numbers → 500 trees vote → disease name**

---

## Complete Pipeline

```
Image (256×256×3) → Grayscale → Resize 128×128 → Extract 7 Features → Random Forest → Disease Label
```

---

## Step 1: The Image

A digital image is a grid of numbers (pixels).

```
Original color image (e.g., 256×256 pixels, 3 channels):

  Each pixel = 3 numbers (Red, Green, Blue)
  Range: 0–255 each
  Total numbers: 256 × 256 × 3 = 196,608 numbers  ← Too many for Random Forest
```

---

## Step 2: Grayscale Conversion

```
Color pixel: (R=120, G=200, B=80)  → 3 numbers
                ↓
Gray pixel:  (147)                  → 1 number

Formula: Gray = 0.299×R + 0.587×G + 0.114×B
```

**Why?** Reduces data by 3×. Disease shows as dark/light spots — color isn't strictly needed for texture patterns.

---

## Step 3: Resize to 128×128

```
Original: 256×256 = 65,536 pixels
Resized:  128×128 = 16,384 pixels  ← 4× smaller, but keeps the pattern
```

**Why?** Makes ALL images the same size, regardless of camera resolution. Every image becomes exactly 16,384 gray numbers.

---

## Step 4: Extract 7 Features (The Core)

From 16,384 pixel values, we calculate **7 summary numbers**:

```
Example pixel values (simplified to 10):
[45, 120, 200, 180, 30, 155, 190, 88, 210, 50]
```

| # | Feature | What It Calculates | Example | What It Tells Us |
|---|---|---|---|---|
| 1 | **Mean** | Sum all pixels ÷ count | 126.8 | Overall brightness. Diseased = darker |
| 2 | **Std Dev** | How spread out pixels are | 66.4 | More spread = more spots |
| 3 | **Min** | Smallest pixel value | 30 | Darkest spot (dead/necrotic tissue) |
| 4 | **Max** | Largest pixel value | 210 | Brightest spot (chlorosis/yellowing) |
| 5 | **Median** | Middle value when sorted | 137.5 | Like mean but ignores extreme spots |
| 6 | **Variance** | Std Dev² | 4,408.96 | Amplified differences |
| 7 | **Entropy** | Randomness/complexity | 3.28 | Higher = more complex = more disease |

### Entropy Explained

```
1. Make a histogram (count how many pixels have each brightness 0–255)
2. Convert counts to probabilities (each count ÷ total pixels)
3. Entropy = −Σ(p × log₂(p))

Healthy leaf:  uniform green   → low entropy  (simple texture)
Diseased leaf: mixed spots     → high entropy  (complex texture)
```

**Result:** 16,384 pixel values → **7 numbers**. This is the compression.

---

## Step 5: YOLO Label Files

The dataset has a `.txt` file for each image:

```
leaf_001.jpg  ←→  leaf_001.txt

Inside leaf_001.txt:
3 0.52 0.48 0.81 0.76
↑  ↑     ↑    ↑    ↑
│  x_center   │  height    ← Bounding box (we IGNORE this)
│       y_center
│            width
│
class_id = 3               ← We USE this → maps to "Tomato_Late_blight"
```

We only read the **class number** for the label. Bounding box coordinates are ignored — Random Forest doesn't need them.

---

## Step 6: Building the Training Table

After processing all images:

```
Image 1 → [142.3, 38.7, 12, 255, 148, 1497.69, 6.82] → "Potato___Early_blight"
Image 2 → [168.1, 22.4, 45, 240, 170,  501.76, 5.91] → "Potato___healthy"
Image 3 → [131.5, 45.2,  8, 248, 125, 2043.04, 7.12] → "Tomato_Late_blight"
...
Image N → [...]                                        → "..."

Final table: N rows × 8 columns (7 features + 1 label)
```

---

## Step 7: Train/Test Split (80/20)

```
Total: 1000 images (example)
  ↓
Train: 800 images → model learns patterns from these
Test:  200 images → model has NEVER seen these, used to test accuracy

stratify=y → ensures each class has equal proportion in both sets
             (e.g., if 20% are Potato_healthy, both sets have ~20%)
```

---

## Step 8: How Random Forest Works

```
               7 features of one image
               [142.3, 38.7, 12, 255, 148, 1497.69, 6.82]
                         ↓
    ┌────────────────────┼────────────────────┐
    ↓                    ↓                    ↓
Tree 1               Tree 2         ...    Tree 500
┌─────────┐         ┌─────────┐          ┌─────────┐
│Variance  │        │Entropy   │         │Mean      │
│> 1000?   │        │> 6.5?   │          │< 130?   │
├──Y───N──┤         ├──Y───N──┤          ├──Y───N──┤
↓         ↓         ↓         ↓          ↓         ↓
...       ...       ...       ...        ...       ...
↓                   ↓                    ↓
"Early              "Late               "Early
 blight"             blight"             blight"

Each tree votes → Majority wins:
  350 trees say "Early_blight"
  150 trees say "Late_blight"
  WINNER: "Early_blight" ✅
```

### Key Hyperparameters:
- **n_estimators=500**: 500 decision trees for robust voting
- **class_weight='balanced'**: Rare classes get more weight so the model doesn't ignore them
- **criterion='gini'**: Measures how pure each tree split is
- **random_state=42**: Ensures reproducibility

---

## Step 9: Results Generated

```
Model predicts all test images → compare with real labels:

                     Predicted
              EB    PH    TEB   TLB   TH
Actual  EB  [ 35    2     1     0     2  ]
        PH  [  1   38     0     0     1  ]
        TEB [  3    0    32     5     0  ]  ← Confusion Matrix
        TLB [  0    1     4    33     2  ]
        TH  [  2    1     0     1    36  ]

Accuracy  = correct ÷ total
Precision = correct per class ÷ total predicted per class
Recall    = correct per class ÷ total actual per class
F1-Score  = harmonic mean of precision and recall
```

---

## Model Details (from leaf_disease_model.pkl)

| Parameter | Value |
|---|---|
| Model Type | RandomForestClassifier |
| Number of Trees | 500 |
| Input Features | 7 |
| Class Weight | balanced |
| Criterion | gini |
| Avg Tree Depth | 22.3 |
| Avg Leaf Nodes | 711 |

### Feature Importances

```
Variance   0.1929  █████████  ← Most important
Std Dev    0.1750  ████████
Min        0.1498  ███████
Median     0.1411  ███████
Max        0.1345  ██████
Entropy    0.1126  █████
Mean       0.0939  ████      ← Least important
```

**Key Finding:** Variance and Std Dev are the most important features — diseased leaves have irregular pixel patterns (spots, lesions) compared to uniform healthy leaves.

---

## Dataset Info

- **Source:** PlantVillage for Object Detection (YOLO format)
- **Link:** https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo
- **5 Classes:** Potato Early Blight, Potato Healthy, Tomato Early Blight, Tomato Late Blight, Tomato Healthy
- **Format:** YOLO annotations (class_id + bounding box per image)

---

## Deployment

```
Trained model (leaf_disease_model.pkl)
        ↓
    Gradio (app.py)
        ↓
    Web page with upload button
        ↓
    User uploads leaf photo → gets disease prediction instantly
        ↓
    Hosted on Hugging Face Spaces (free, public URL)
```
