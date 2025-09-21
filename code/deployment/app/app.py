import json
import os
from pathlib import Path
import requests
import streamlit as st

MODEL_DIR = Path("/models")
META_PATH = MODEL_DIR / "meta.json"

meta = json.loads(META_PATH.read_text())
FEATURES = meta["feature_names"]
TARGETS  = meta["target_names"]

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.title("Breast Cancer Classifier")

with st.sidebar:
    st.subheader("API endpoint")
    st.code(API_URL)
    if st.button("Health check"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(str(e))

st.markdown("### Enter features")
cols = st.columns(2)
values = {}
for i, f in enumerate(FEATURES):
    with cols[i % 2]:
        values[f] = st.number_input(f.replace("_", " ").title(), value=0.0, step=0.1)

if st.button("Predict"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"features": values}, timeout=10)
        if r.ok:
            out = r.json()
            st.success(f"Prediction: **{out['predicted_class']}** (class {out['class_index']})")
            st.json(out["proba"])
        else:
            st.error(r.text)
    except Exception as e:
        st.error(str(e))
