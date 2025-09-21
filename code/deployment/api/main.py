from pathlib import Path
from typing import Dict, List
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Breast Cancer Classifier API")

MODEL_DIR = Path("/models")
MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH  = MODEL_DIR / "meta.json"

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
FEATURES: List[str] = meta["feature_names"]
TARGETS:  List[str] = meta["target_names"]  # ["malignant", "benign"]

class PredictRequest(BaseModel):
    features: Dict[str, float]

class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    proba: Dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES, "targets": TARGETS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    missing = [f for f in FEATURES if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    x = [[req.features[f] for f in FEATURES]]
    probs = model.predict_proba(x)[0]
    idx = int(probs.argmax())
    return PredictResponse(
        predicted_class=TARGETS[idx],
        class_index=idx,
        proba={TARGETS[i]: float(p) for i, p in enumerate(probs)},
    )
