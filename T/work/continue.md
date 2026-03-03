# Future Work & Post-Testing Updates

## Experiments Status

### Experiment 3: `sage_nopca_bestresults.ipynb` — DONE
- **What:** HOG (16x16) + RF with `max_depth=None` (unlimited tree depth)
- **Instance:** SageMaker ml.c5.4xlarge (16 vCPU, 32 GB RAM) — same as Exp 2 and 3a
- **Result:** 75.31% accuracy, 511.1 MB model size
- **Results file:** `T/exp3/b/sage_nopca_bestresults.ipynb`
- **Key finding:** Removing depth cap gave NO improvement over depth=25 (75.52%). Proves ~75% is the **feature ceiling** for HOG + color stats on 38 classes. Deeper trees just overfit.
- **T_WORK.md:** Updated with full per-class results

### Experiment 4: `sage_mobilenetresult.ipynb` — COMPLETED
- **What:** MobileNetV3-Small with transfer learning (end-to-end deep learning)
- **Status:** Successfully executed with normalization fix, learning rate adjustment, and data augmentation.
- **Actual Results:** 98.13% accuracy, 1.1 MB (.tflite), 8.7 MB (.h5).
- **Instance:** ml.g4dn.xlarge ($0.53/hr, 1x T4 GPU, 16GB GPU RAM)
- **Note:** Transfer learning successfully overcame the 75% feature ceiling found in traditional ML approaches.

---

## After Both Experiments Complete — DONE

### 1. Update T_WORK.md — DONE
Filled in the actual numbers for Experiments 3 and 4:
- Exact accuracy percentages (75.31% for 3b, 98.13% for 4)
- Exact model file sizes (511.1 MB for 3b, 1.1 MB .tflite / 8.7 MB .h5 for 4)
- Full 38-class precision/recall/F1 tables for all experiments
- Potato_healthy recovered to 93% recall in Exp 4 (was 0% in Exp 1-3b)

### 2. Final Comparison Table — DONE

```
| # | Method                        | Accuracy | Model Size |
|---|-------------------------------|----------|------------|
| 1 | 7 grayscale stats + RF        | 44.48%   | 699.8 MB   |
| 2 | HOG 8x8 + PCA + RF            | 53.55%   | —          |
| 3a| HOG 16x16 + RF (depth=25)     | 75.52%   | 445.6 MB   |
| 3b| HOG 16x16 + RF (depth=None)   | 75.31%   | 511.1 MB   |
| 4 | MobileNetV3-Small (fine-tuned) | 98.13%   | 1.1 MB     |
```

### 3. Discussion Points to Write

Based on results, write about:
- **The accuracy ceiling of hand-crafted features is ~75%.** Removing depth cap (3a→3b) gave zero improvement. This is the hard limit of HOG+RF on 38 classes.
- **The PCA lesson:** 53% → 76% from removing PCA. Document this as a warning for other researchers.
- **Edge deployment tradeoff:** RF is large (445-511 MB) but needs only CPU. MobileNet is tiny (1.1 MB .tflite) but needs TFLite runtime.
- **Class imbalance impact:** Potato_healthy (30 test samples) went from 0% recall (Exp 1-3b) to 93% recall (Exp 4) — CNN features overcame the imbalance.

### 4. Deployment Code (If Needed Later)
After the paper experiments, if a web app or mobile demo is needed:
- **For RF model:** Use the `extract_features()` function + `joblib.load()` pipeline
- **For MobileNet:** Use the `.tflite` file with TFLite interpreter
- The web app code in `vishnu/` folder may need updating to match whichever final model is chosen

---

## Potential Issues During Execution

### sage_nopca_bestresults.ipynb — COMPLETED
- **Actual result:** 511.1 MB model, 75.31% accuracy. No issues during execution.

### sage_mobilenetresult.ipynb
- **Risk 1:** TensorFlow not installed on SageMaker image. Fix: first cell runs `pip install tensorflow`.
- **Risk 2:** No GPU available. Training will be slow on CPU (~2-5 hours for 20 epochs). On GPU it's ~10-20 minutes.
- **Memory:** Now uses tf.data batch loading — loads 32 images at a time, NOT all into memory. Safe on any instance.
- **Instance:** ml.g4dn.xlarge recommended ($0.53/hr, 1x T4 GPU)

---

## Optional Future Extensions (Not Required for Current Paper)

- **MobileNetV3-Large vs Small comparison** — run both and add to paper as 4a/4b
- **Hybrid CNN+RF experiment** — use MobileNet as feature extractor + RF classifier (discussed in comparison.md, decided against for deployment reasons)
- **TFLite quantization** — INT8 quantization to shrink MobileNet further for microcontrollers
- **Real-world field testing** — test with actual phone camera photos, not just PlantVillage dataset images
- **Web/mobile app demo** — deploy best model as a working prototype
