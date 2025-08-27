
import streamlit as st
import numpy as np
import cv2
import json
import hashlib
import os
from datetime import datetime
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib

st.set_page_config(page_title="NeuroChainAI", layout="centered")
st.title("🧠 Neurological Disorder Prediction")
st.subheader("Upload a Brain Scan Image")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 0)
    resized = cv2.resize(image, (64, 64)).reshape(1, 64, 64, 1) / 255.0

    cnn_model = load_model("cnn_model.keras")
    from tensorflow.keras import Sequential
    cnn_feature_model = Sequential(cnn_model.layers[:-2])
    features = cnn_feature_model.predict(resized)

    gru_model = load_model("gru_model.keras")
    gru_pred = gru_model.predict(features.reshape(1, 1, features.shape[1]))[0]

    xgb_model = joblib.load("xgb_model.pkl")
    xgb_pred_label = xgb_model.predict(features)[0]
    xgb_pred_onehot = np.zeros(5)
    xgb_pred_onehot[xgb_pred_label] = 1

    final_pred = (0.4 * gru_pred) + (0.6 * xgb_pred_onehot)
    labels = [
        "Alzheimer's likelihood",
        "Parkinson's likelihood",
        "Stroke risk indicator",
        "Brain tumor indicator",
        "Normal / other"
    ]
    result = dict(zip(labels, final_pred.tolist()))

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    file_sha = hashlib.sha256(json.dumps(result).encode()).hexdigest()
    record_hash = hashlib.sha256((file_sha + timestamp).encode()).hexdigest()

    ledger = {
        "Record hash": record_hash,
        "Prev hash": "None (genesis)",
        "File SHA-256": file_sha,
        "Model": "cnn-gru-xgb-v1.0",
        "Timestamp": timestamp
    }

    st.image(image, caption="Uploaded Scan", width=300)
    st.markdown("## 🧪 Predicted Possibilities")
    for key, val in result.items():
        st.markdown(f"- **{key}**: {val*100:.1f}%")

    st.markdown("## 🧾 Ledger Receipt")
    for key, val in ledger.items():
        st.markdown(f"- **{key}**: `{val}`")

    st.info("These percentages are illustrative only. A licensed clinician must review real scans.")
    st.markdown("###### DISCLAIMER: Demo results are not diagnostic. For concerns, consult a neurologist.")

    os.makedirs("ledger", exist_ok=True)
    with open("ledger/ledger_store.jsonl", "a") as f:
        json.dump({**ledger, **result}, f)
        f.write("\n")
