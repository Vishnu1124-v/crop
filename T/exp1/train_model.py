"""
train_model.py — Train Random Forest on PlantVillage dataset

Usage:
    python train_model.py --dataset_path ./PlantVillage

The dataset_path should contain 5 subfolders:
    Potato___Early_blight/
    Potato___healthy/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___healthy/

Each subfolder contains .jpg or .png leaf images.
"""

import os
import argparse
import cv2
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


CLASS_NAMES = [
    "Potato__Early_blight",
    "Potato__healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
]


def extract_features(image):
    """Extract 7 statistical features from a leaf image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def load_dataset(dataset_path):
    """Load images from class subfolders and extract features."""
    features = []
    labels = []
    print("Loading images and extracting features...")

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Match folder name to a class
        matched_class = None
        for cls in CLASS_NAMES:
            if cls.lower().replace("__", "_") in folder_name.lower().replace("___", "_"):
                matched_class = cls
                break
        if matched_class is None:
            print(f"  ⚠ Skipping unrecognized folder: {folder_name}")
            continue

        count = 0
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            feat = extract_features(image)
            features.append(feat)
            labels.append(matched_class)
            count += 1

        print(f"  ✅ {matched_class}: {count} images")

    return np.array(features), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest on PlantVillage")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset folder containing class subfolders",
    )
    args = parser.parse_args()

    # Load data
    X, y = load_dataset(args.dataset_path)
    print(f"\nTotal samples: {len(X)}")

    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train Random Forest
    print("\nTraining Random Forest (n_estimators=500, class_weight='balanced')...")
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  ACCURACY: {accuracy:.2%}")
    print(f"{'='*50}")

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n")
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.2%}\n\n")
        f.write(report)
    print("  → Saved classification_report.txt")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix — Random Forest (7 Statistical Features)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  → Saved confusion_matrix.png")

    # Feature Importance
    feature_names = ["Mean", "Std Dev", "Min", "Max", "Median", "Variance", "Entropy"]
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_names, importances, color="#2ecc71")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("  → Saved feature_importance.png")

    # Save model
    joblib.dump(model, "leaf_disease_model.pkl")
    print("  → Saved leaf_disease_model.pkl")

    print("\n✅ Done! Use these files in your paper:")
    print("   confusion_matrix.png")
    print("   feature_importance.png")
    print("   classification_report.txt")
    print("   leaf_disease_model.pkl")


if __name__ == "__main__":
    main()
