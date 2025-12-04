import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# -----------------------------
# Load Model
# -----------------------------
with open("newmodel.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Load Breast Cancer Dataset for column names
# -----------------------------
cancer = load_breast_cancer(as_frame=True)
feature_names = cancer.data.columns.tolist()

st.title("Breast Cancer Prediction App")
st.write("ML Workflow Demo using LinearSVC Model")

# ---------------------------------------------------------
# Sidebar: Prediction Options
# ---------------------------------------------------------
mode = st.sidebar.radio("Select Input Mode",
                        ["Manual Input", "Upload CSV"])

# ---------------------------------------------------------
# Manual Form Input
# ---------------------------------------------------------
if mode == "Manual Input":
    st.subheader("Enter Feature Values")

    user_input = {}
    cols = st.columns(3)

    for i, col in enumerate(feature_names):
        with cols[i % 3]:
            user_input[col] = st.number_input(
                col, 
                value=float(cancer.data[col].mean()),
                format="%.5f"
            )

    # Convert to DataFrame for prediction
    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        result = "Malignant (Cancer Detected)" if pred == 0 else "Benign (Non-Cancer)"
        st.success(f"Prediction: **{result}**")

# ---------------------------------------------------------
# CSV Upload for Batch Prediction
# ---------------------------------------------------------
else:
    st.subheader("Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV containing the 30 features", type=["csv"])

    if file:
        data = pd.read_csv(file)

        st.write("Uploaded Data:")
        st.dataframe(data.head())

        # Predict
        preds = model.predict(data)

        data["Prediction"] = ["Malignant" if p == 0 else "Benign" for p in preds]

        st.success("Prediction Complete")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv, "results.csv")

# ---------------------------------------------------------
# Show Dataset Information
# ---------------------------------------------------------
st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"Total Features: {len(feature_names)}")
st.sidebar.write("Model Used: LinearSVC (C=10)")
