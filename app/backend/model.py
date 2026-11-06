import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pickle
import lightgbm as lgb
from huggingface_hub import hf_hub_download
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_MODEL_NAME = "distilbert-base-uncased"
IMAGE_MODEL_NAME = "efficientnet_b0"
HF_REPO_ID = "bboy-07/smart-product-pricing-models"
IMG_SIZE = 224

class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        self.image_model = timm.create_model(IMAGE_MODEL_NAME, pretrained=False, num_classes=0)
        text_features = self.text_model.config.hidden_size        # 768
        image_features = self.image_model.num_features            # 1280
        self.head = nn.Sequential(
            nn.Linear(text_features + image_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def extract_embeddings(self, input_ids, attention_mask, image):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = text_output.last_hidden_state[:, 0, :]
        image_embedding = self.image_model(image)
        return torch.cat([text_embedding, image_embedding], dim=1)


def load_models():
    # Download models from HF
    nn_path = hf_hub_download(repo_id=HF_REPO_ID, filename="nn_model.pt", repo_type="model")
    pca_path = hf_hub_download(repo_id=HF_REPO_ID, filename="pca.pkl", repo_type="model")
    lgbm_path = hf_hub_download(repo_id=HF_REPO_ID, filename="lgbm_model.txt", repo_type="model")

    # Load NN
    nn_model = MultiModalModel().to(DEVICE)
    nn_model.load_state_dict(torch.load(nn_path, map_location=DEVICE))
    nn_model.eval()

    # Load PCA
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    # Load LightGBM
    lgbm_model = lgb.Booster(model_file=lgbm_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    return nn_model, pca, lgbm_model, tokenizer


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def predict(image: Image.Image, text: str, nn_model, pca, lgbm_model, tokenizer) -> float:
    # Preprocess
    image_tensor = preprocess_image(image)
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Extract embeddings
    with torch.no_grad():
        embedding = nn_model.extract_embeddings(input_ids, attention_mask, image_tensor)
        embedding = embedding.cpu().numpy()

    # PCA + LightGBM
    embedding_pca = pca.transform(embedding)
    log_price = lgbm_model.predict(embedding_pca)[0]
    price = float(np.expm1(log_price))
    return max(0.0, price)
