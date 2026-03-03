# The Two Methodologies in `T/`

This folder now contains two completely distinct approaches to the Random Forest training so they don't get mixed up.

## 1. `old_7_feature_method/`
*   **The Approach:** Extracted only 7 mathematical features from strictly *grayscale* images (Mean, Std, Min, Max, Median, Variance, Entropy).
*   **The Algorithm:** Random Forest Classifier (500 Trees).
*   **The Accuracy:** ~45%.
*   **The File Sizes:** Because it struggled to categorize 38 diseases using only 7 numbers from black and white images, the trees grew incredibly deep and massive. The resulting model was huge.
    *   `leaf_disease_model_CLEAN.pkl` is **~700 MB**.
    *   `leaf_disease_model_TINY.pkl` (Heavy compression version) is **~310 MB**.

## 2. `new_hog_pca_method/`
*   **The Approach:** Extracts **13 features** (RGB + Grayscale), **Gaussian Blur** to remove noise, and **HOG** (Histogram of Oriented Gradients) to learn the physical textures and shapes of the spots.
*   **The Compression:** Because HOG creates thousands of data points, it uses **PCA (Principal Component Analysis)** to zip the features together before sending them to the Random Forest.
*   **The Algorithm:** Random Forest Classifier (500 Trees, `max_depth=25`, `min_samples_leaf=2`). Limiting the depth keeps the model small.
*   **The expected Accuracy:** ~80-90%+.
*   **The expected File Size:** The `max_depth` restriction + PCA compression will yield a highly deployable ~50MB `leaf_disease_pipeline.pkl`. All code resides in `colab_train_notebook.ipynb`.
