# 🛒 Smart Product Pricing — Multimodal E-Commerce Price Prediction

> Predicting product prices from images and text using a multimodal deep learning architecture combined with gradient boosting.

---

## 🖥️ Live Demo Screenshots

**Prediction 1 — Rani 14-Spice Mango Chutney (10.5 oz)**
![Mango Chutney Prediction](OUTPUTS/s1.png)

**Prediction 2 — Aleias Gluten Free Italian Breadcrumbs (Pack of 12)**
![Breadcrumbs Prediction](OUTPUTS/Screenshot%202026-11-21%20153021.png)

Both predictions running live on EC2 (`16.16.26.32:8000`) — full multimodal inference (DistilBERT + EfficientNet + PCA + LightGBM) completing in under 3 seconds on CPU.

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

## ❓ But We Already Know Prices — So Why Predict?

This is the most important question to answer before understanding this project.

**The key insight: we know prices of existing products, but not new ones.**

The model learns patterns from millions of existing listings (image + text → price) and uses that knowledge to **suggest a fair price for something it has never seen before.**

> Think of it like a real estate agent — they can estimate the price of a house they've never seen before just by looking at photos and reading the description, because they've seen thousands of similar houses. This model does exactly that, but for products.

### Real-World Use Cases This Unlocks

| Use Case | How the Model Helps |
|---|---|
| 🆕 New seller listing a product | No historical price exists → model suggests a fair price range |
| 🚨 Fraud / anomaly detection | Seller lists "iPhone 15 Pro" at ₹500 → model predicts ₹80,000 → flag it |
| 🔄 Dynamic repricing | Seller updates image/description → model re-evaluates fair price |
| 🗂️ Catalog quality control | Predicted vs listed price diverges wildly → wrong category, fake image, or misleading text |
| 📊 New product launch | No market data yet → model estimates based on visual/textual similarity to existing products |

So the training data (known prices) is just the **teacher**. The deployed model serves people dealing with **unknown prices** — which is the actual business problem.

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

## 🚀 Deployment

### Infrastructure — AWS EC2

The app runs on an **AWS EC2 m7i-flex.large** instance (8 GB RAM, 2 vCPU) on Amazon Linux 2023.

**Why m7i-flex.large:**
The model stack at inference time loads DistilBERT (~250 MB), EfficientNet-B0 (~20 MB), PyTorch and all dependencies (~400 MB), and the OS/Python process (~200 MB) — totalling under 1 GB RAM. The t3.micro (1 GB RAM) couldn't hold this stack at startup and threw an out-of-memory error. The m7i-flex.large gives 8 GB RAM with 7+ GB headroom, which keeps inference stable.

**Storage — Separate EBS Volume:**
The root volume (8 GB) is left for the OS only. A separate 20 GB EBS gp3 volume is attached and mounted at `/data`. Everything heavy — the Python venv, PyTorch (~700 MB), all pip packages, and the HF model cache — lives on `/data`. This keeps the root volume from filling up and makes it easy to detach/snapshot the data volume independently.

```
/           →  8 GB root   →  Amazon Linux OS only
/data       →  20 GB EBS   →  venv, pip packages, HF cache, app code
```

Volume UUID: `a4b6c4cd-9dd7-4238-9b40-13f9ed467c9b` (mounted via `/etc/fstab` with `nofail`)

**HF model cache** is redirected from the default `~/.cache/huggingface` to `/data/hf-cache` via the `HF_HOME` environment variable so model weights (`nn_model.pt`, `pca.pkl`, `lgbm_model.txt`) download to the EBS volume on first startup.

---

### Why Not Hugging Face Spaces

HF Spaces was the first option considered since the models are already hosted at `bboy-07/smart-product-pricing-models`. Two deployment paths were explored:

**Option 1 — HF Serverless Inference API:**
HF hosts base models like `distilbert-base-uncased` on a free serverless tier. However, the inference pipeline uses a **custom fine-tuned `nn_model.pt`** whose weights are not the base DistilBERT weights — they've been jointly trained end-to-end for price prediction. The serverless API only serves standard Hub models, not custom `.pt` files, so this path doesn't work for this pipeline.

**Option 2 — HF Gradio / Docker Space:**
A Docker Space exposing a FastAPI endpoint (with the full inference stack running on HF's free 16 GB CPU) would have worked technically. However, as of 2025, HF requires a **paid PRO plan** to create Gradio or Docker Spaces on personal accounts. Free accounts are limited to static HTML Spaces and up to 2 ZeroGPU Spaces. Neither fits this use case.

Since both free HF paths were blocked, the decision was to run inference directly on EC2 with a large enough instance.

---

### EC2 Setup — Step by Step

**Security group inbound rules:**

| Port | Purpose |
|---|---|
| 22 | SSH access |
| 80 | HTTP |
| 8000 | FastAPI / uvicorn |

**1. Format and mount EBS volume:**
```bash
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir /data
sudo mount /dev/nvme1n1 /data
sudo chown ec2-user:ec2-user /data
```

`lsblk` confirmed `nvme0n1` (8 GB) as root and `nvme1n1` (20 GB) as the attached EBS volume.

**2. Auto-mount on reboot via fstab:**
```bash
sudo blkid /dev/nvme1n1
# UUID="a4b6c4cd-9dd7-4238-9b40-13f9ed467c9b"

echo 'UUID=a4b6c4cd-9dd7-4238-9b40-13f9ed467c9b  /data  ext4  defaults,nofail  0  2' | sudo tee -a /etc/fstab
sudo mount -a
```

**3. System packages:**
```bash
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-devel python3-pip git gcc tmux
```

**4. Set HF model cache to EBS volume:**
```bash
echo 'export HF_HOME=/data/hf-cache' >> ~/.bashrc
source ~/.bashrc
mkdir -p /data/hf-cache
```

**5. Clone production branch:**
```bash
git clone -b production https://github.com/DarshanBhabad/Smart-Product-Pricing.git /data/smartprice
```

**6. Create venv on EBS and install dependencies:**
```bash
python3.11 -m venv /data/venv
source /data/venv/bin/activate
cd /data/smartprice/app/backend
pip install --cache-dir /data/pip-cache -r requirements.txt
```

> **NumPy compatibility fix:** PyTorch 2.1.0 is incompatible with NumPy 2.x. pip resolves to NumPy 2.x by default on a fresh install, which causes a `RuntimeError: Numpy is not available` at inference time. Fix:
> ```bash
> pip install "numpy<2.0"
> ```

**7. Run the backend persistently with tmux:**

The app is served by uvicorn on port 8000. tmux keeps it running after the SSH session closes.

```bash
tmux new -s smartprice
source /data/venv/bin/activate
cd /data/smartprice/app/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first startup, `load_models()` downloads `nn_model.pt`, `pca.pkl`, and `lgbm_model.txt` from HF Hub into `/data/hf-cache`. Once you see:

```
Loading models...
Models loaded!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Detach from tmux with `Ctrl+B` then `D` — the app keeps running in the background.

**8. Frontend — served by FastAPI:**

The frontend (`app/frontend/index.html`) is served as a static file directly by the FastAPI backend via:

```python
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("../frontend/index.html")
```

So there is no separate frontend server or tmux session needed. Everything is served from port 8000. The `BACKEND_URL` in `index.html` is set to `http://16.16.26.32:8000` to point API calls to the same EC2 instance.

---

### Useful Commands

```bash
# SSH into instance
ssh -i smartprice-key.pem ec2-user@16.16.26.32

# Re-attach to running app
tmux attach -t smartprice

# Check RAM usage
free -h

# Check disk usage
df -h

# Confirm app is listening on port 8000
sudo ss -tlnp | grep 8000

# Pull latest code and restart
cd /data/smartprice && git pull origin production
tmux attach -t smartprice
# Ctrl+C to stop, then:
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

### Disk Usage Breakdown

| Path | Contents | Size |
|---|---|---|
| `/` | Amazon Linux OS | ~2–3 GB |
| `/data/venv` | Python venv + all packages incl. PyTorch | ~2 GB |
| `/data/hf-cache` | DistilBERT + model weights from HF Hub | ~500 MB |
| `/data/smartprice` | App code (cloned from GitHub) | ~few MB |
| `/data/pip-cache` | pip download cache (safe to delete) | ~500 MB |

---

## 👤 Author

**Darshan Bhabad**
- GitHub: [@DarshanBhabad](https://github.com/DarshanBhabad)
- Kaggle: [@bboyattitude](https://www.kaggle.com/bboyattitude)
- Built for: Mavericks Amazon ML Challenge — Kaggle
