from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io
from model import load_models, predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
print("Loading models...")
nn_model, pca, lgbm_model, tokenizer = load_models()
print("Models loaded!")

# Serve frontend
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("../frontend/index.html")

@app.post("/predict")
async def predict_price(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    img = Image.open(io.BytesIO(await image.read()))
    price = predict(img, text, nn_model, pca, lgbm_model, tokenizer)
    return {
        "predicted_price": round(price, 2),
        "price_range": {
            "low": round(price * 0.85, 2),
            "high": round(price * 1.15, 2)
        }
    }
