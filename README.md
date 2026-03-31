# 🛒 Smart Product Pricing — Multimodal E-Commerce Price Prediction

> Predicting product prices from images and text using a multimodal deep learning architecture combined with gradient boosting.

---

## 📊 Results Summary

| Approach | Architecture | SMAPE |
|---|---|---|
| Baseline | Multimodal NN + Dense Head | 46.1% |
| Iteration 2 | Multimodal NN + LightGBM | ~31% |
| **Final** | **Multimodal NN + PCA + LightGBM** | **19.41%** |

---

## 🧠 Problem Statement

E-commerce platforms like Amazon deal with **millions of product listings daily**. When a new seller lists a product, they must manually set a price — often with no reference point. This leads to:

- Underpricing → seller loses money
- Overpricing → no sales
- Fraudulent listings (e.g. fake iPhone listed at ₹500)

The goal of this project is to **automatically predict the price of a product** given only its **product image** and **text description** — the same information a seller provides when creating a listing.

This was built for the **Mavericks Amazon ML Challenge** on Kaggle.

---

## 📁 Repository Structure

```
├── code0.zip                        # Approach 1: Baseline NN with Dense Head (SMAPE 46.1%)
├── lightGM_1_output.zip             # Approach 2: NN + LightGBM without PCA (SMAPE ~31%)
├── FINAL_SPP_BOOSTING_PCA.ipynb     # Approach 3: NN + PCA + LightGBM — Final Solution (SMAPE 18%)
├── lgbm_model.txt                   # Saved LightGBM model (final)
├── nn_model.pt                      # Saved Neural Network weights (final)
├── pca.pkl                          # Saved PCA transformer (final)
├── submission.csv                   # Final submission file
├── X_val.npy                        # Validation embeddings (saved to avoid recompute)
├── y_val.npy                        # Validation targets (saved to avoid recompute)
└── README.md
```

---

## 🏗️ Architecture Overview

### Why Multimodal?

Product price is determined by **both** visual and textual signals:

| Signal | Example | What It Tells the Model |
|---|---|---|
| Text | "pure silk, handwoven, banarasi" | High-end product → higher price |
| Image | Shows cheap stitching, plain fabric | Contradicts text → lower price |
| Both together | Full context | Much more accurate than either alone |

A seller can lie in the text — but the image doesn't lie. The multimodal approach catches this.

---

## 🔁 Evolution of Approaches

### Approach 1 — Baseline: Multimodal NN with Dense Head (`code0.zip`)
**SMAPE: 46.1%**

```
Images → EfficientNet-B0 ──┐
                            ├──► Combined Embeddings (2048 dims) ──► Dense Head ──► Price
Text   → DistilBERT    ────┘
```

**Architecture:**
- **Text encoder:** DistilBERT (`distilbert-base-uncased`) — extracts 768-dim CLS token embedding
- **Image encoder:** EfficientNet-B0 (pretrained on ImageNet) — extracts 1280-dim global embedding
- **Fusion:** Concatenation → 2048-dim combined vector
- **Prediction head:** `Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→1)`
- **Target:** `log1p(price)` to handle price skewness; predictions converted back with `expm1`
- **Loss:** MSELoss | **Optimizer:** AdamW | **Epochs:** 10

**Limitation:** The dense head is a generic MLP. It doesn't capture complex feature interactions between embedding dimensions as well as tree-based models.

---

### Approach 2 — Two-Stage: NN + LightGBM (`lightGM_1_output.zip`)
**SMAPE: ~31%**

```
STAGE 1 — Make embeddings price-aware:
Images → EfficientNet ──┐
                         ├──► Embeddings ──► Dense Head ──► Price  (trains jointly)
Text   → DistilBERT  ───┘         ↑ backprop flows all the way back

STAGE 2 — Freeze NN, use LightGBM for final prediction:
Images → EfficientNet ──┐
                         ├──► Frozen Embeddings ──► LightGBM ──► Price ✅
Text   → DistilBERT  ───┘         (Dense Head discarded)
```

**Key insight — Joint Training:**
When the dense head trains end-to-end, the error signal backpropagates through the entire network — forcing EfficientNet and DistilBERT to produce embeddings that are specifically useful for price prediction. These are called **price-aware embeddings**.

After training, the dense head is **discarded**. The frozen NN acts purely as a feature extractor, and LightGBM handles the final regression.

**Why LightGBM over Dense Head:**
- Captures complex non-linear feature interactions via tree splits
- Naturally handles tabular-style feature vectors
- Built-in regularization, feature importance, early stopping
- More robust to high-dimensional inputs

**Limitation observed:** Large train/validation RMSE gap indicated overfitting:
```
Round 1000:  train RMSE = 0.046   valid RMSE = 0.225
```
Root cause: LightGBM was finding spurious patterns across all 2048 embedding dimensions.

---

### Approach 3 — Final: NN + PCA + LightGBM (`FINAL_SPP_BOOSTING_PCA.ipynb`)
**SMAPE: 19.41%** ✅

```
STAGE 1 — Joint NN training (same as Approach 2)

STAGE 2 — PCA compression + LightGBM:
Frozen Embeddings (2048 dims)
        ↓
      PCA                          ← NEW
        ↓
Compressed Embeddings (300 dims)
        ↓
    LightGBM ──► Final Price ✅
```

**Why PCA fixed the overfitting:**

The 2048-dimensional embedding space contains many noisy, redundant dimensions. LightGBM was exploiting spurious correlations in those noise dimensions — patterns that existed in training data but not in real data.

PCA projects the embeddings onto the **300 most meaningful directions** (those explaining the most variance), discarding the remaining 1748 noisy dimensions. This forces LightGBM to learn from signal, not noise.

```
Without PCA:   train RMSE = 0.046   valid RMSE = 0.225   (5x gap → overfitting)
With PCA:      train/valid gap closes significantly → SMAPE drops from 31% to 18%
```

**Anti-overfitting LightGBM configuration:**
```python
LGBM_PARAMS = {
    'learning_rate': 0.03,       # slower learning → less memorization
    'num_leaves': 63,            # simpler trees (was 127)
    'min_data_in_leaf': 50,      # no splits on tiny groups
    'feature_fraction': 0.6,     # each tree sees 60% of dims (like dropout)
    'bagging_fraction': 0.7,     # each tree sees 70% of rows
    'lambda_l1': 0.1,            # L1 regularization
    'lambda_l2': 0.1,            # L2 regularization
}
LGBM_NUM_ROUNDS = 2000
LGBM_EARLY_STOPPING = 100
```

---

## 📐 Evaluation Metric — SMAPE

**Symmetric Mean Absolute Percentage Error:**

```
SMAPE = (1/n) × Σ [ 200 × |predicted - actual| / (|actual| + |predicted|) ]
```

SMAPE is used because it treats overestimation and underestimation symmetrically and handles the wide price range in e-commerce naturally.

| SMAPE | Real World Meaning |
|---|---|
| 46.1% | Off by ₹461 on a ₹1000 product |
| 31% | Off by ₹310 on a ₹1000 product |
| **18%** | **Off by ₹180 on a ₹1000 product** ✅ |

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Text Encoder | DistilBERT (`distilbert-base-uncased`) via HuggingFace Transformers |
| Image Encoder | EfficientNet-B0 via `timm` |
| Deep Learning | PyTorch |
| Gradient Boosting | LightGBM |
| Dimensionality Reduction | scikit-learn PCA |
| Data Processing | Pandas, NumPy, PIL |
| Training Environment | Kaggle (GPU — CUDA) |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install transformers lightgbm timm torch torchvision scikit-learn pandas numpy tqdm pillow requests
```

### Training (Final Approach)
Open `FINAL_SPP_BOOSTING_PCA.ipynb` on Kaggle and run all cells. The pipeline:

1. Downloads product images from URLs in the CSV
2. Trains the multimodal NN end-to-end for 10 epochs (Stage 1)
3. Freezes NN, extracts embeddings for all training samples
4. Applies PCA: 2048 → 300 dimensions
5. Trains LightGBM on compressed embeddings with early stopping
6. Extracts test embeddings → PCA → LightGBM → final predictions
7. Saves `submission.csv`

### Loading Saved Models (skip retraining)
```python
import pickle, lightgbm as lgb, torch

lgbm_model = lgb.Booster(model_file='lgbm_model.txt')

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

model = MultiModalModel().to(device)
model.load_state_dict(torch.load('nn_model.pt'))
model.eval()
```

---

## 💡 Key Learnings

1. **Joint training matters** — Training the NN with a dense head first makes embeddings price-aware before handing them to LightGBM. Skipping this step would give LightGBM generic, uninformed features.

2. **High-dimensional embeddings need compression** — Raw 2048-dim embeddings give LightGBM too many dimensions to find noise patterns in. PCA to 300 dims was the single biggest performance jump.

3. **SMAPE penalizes cheap items more** — A ₹100 prediction error on a ₹200 item contributes far more to SMAPE than the same error on a ₹5000 item. `log1p` target transformation helps balance this.

4. **Save models immediately after training** — Kaggle sessions expire. Always save `lgbm_model.txt`, `nn_model.pt`, and `pca.pkl` right after training completes, before any subsequent code can fail.

5. **LightGBM early stopping needs a clean validation split** — Use a fixed non-shuffled 90/10 split so val arrays can be exactly recreated if needed.

---

## 📈 Further Improvements

- Swap DistilBERT for `roberta-base` or a product-specific language model
- Add structured features (category, brand) directly into LightGBM alongside embeddings
- Try ensemble of dense head output + LightGBM output with learned weights
- Use cross-validation instead of a single 90/10 split for more robust LightGBM training

---

## 🔗 Links

| Resource | Link |
|---|---|
| 📦 Dataset | [Mavericks Amazon ML Dataset](https://www.kaggle.com/datasets/bboyattitude/mavericks-amazon-ml-dataset) |
| 📓 Kaggle Notebook | [ecommerce-using-boosting](https://www.kaggle.com/code/bboyattitude/ecommerce-using-boosting?scriptVersionId=306863797) |
| 💻 GitHub | [DarshanBhabad](https://github.com/DarshanBhabad) |

---

## 👤 Author

**Darshan Bhabad**
- GitHub: [@DarshanBhabad](https://github.com/DarshanBhabad)
- Kaggle: [@bboyattitude](https://www.kaggle.com/bboyattitude)
- Built for: Mavericks Amazon ML Challenge — Kaggle
