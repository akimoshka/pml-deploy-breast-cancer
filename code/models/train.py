import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]  # repo root
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    # 30 features
    feature_names = list(data.feature_names)
    # "malignant" or "benign"
    target_names = list(data.target_names)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
    )
    pipe.fit(Xtr, ytr)

    # save model + meta
    joblib.dump(pipe, MODELS_DIR / "model.pkl")
    (MODELS_DIR / "meta.json").write_text(
        json.dumps({"feature_names": feature_names, "target_names": target_names}, indent=2)
    )
    print("Saved models/model.pkl and models/meta.json")

if __name__ == "__main__":
    main()