import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("creditcard.csv")
    except:
        uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            return None
    return df

df = load_data()
if df is None:
    st.warning("Please upload the dataset to continue.")
    st.stop()

# -------------------------------
# Train and Save/Load Model
# -------------------------------
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        X = df.drop("Class", axis=1)
        y = df["Class"]
        X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    return model

model = load_or_train_model()

# Prepare Data for Evaluation
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🔎 Navigation")
section = st.sidebar.radio("Go to", [
    "Dataset Overview", "Summary Statistics", "EDA", "Live Prediction", "Model Evaluation", "Conclusion"
])

# -------------------------------
# Dataset Overview
# -------------------------------
if section == "Dataset Overview":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("""
    This app analyzes credit card transactions to detect fraud using XGBoost.
    """)
    st.subheader("Sample Data")
    st.dataframe(df.head())

# -------------------------------
# Summary Statistics
# -------------------------------
elif section == "Summary Statistics":
    st.title("📋 Summary Statistics")
    st.subheader("Data Description")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.table(df.isnull().sum())

# -------------------------------
# EDA
# -------------------------------
elif section == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Class', data=df, palette='Set2', ax=ax1)
    ax1.set_title("0 = Legit, 1 = Fraud")
    st.pyplot(fig1)

    st.subheader("Transaction Amount by Class")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Distribution of V1 for Each Class")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=df, x='V1', hue='Class', bins=50, kde=True, ax=ax4)
    st.pyplot(fig4)

    st.subheader("Fraud by Hour")
    df['Hour'] = (df['Time'] // 3600) % 24
    fig5, ax5 = plt.subplots()
    sns.histplot(data=df[df['Class'] == 1], x='Hour', bins=24, ax=ax5, color="red")
    st.pyplot(fig5)

# -------------------------------
# Live Prediction
# -------------------------------
elif section == "Live Prediction":
    st.title("🧠 Live Transaction Prediction")
    st.markdown("Enter transaction values below to predict fraud status:")

    user_input = {}
    for col in X.columns:
        val = st.number_input(f"{col}", value=0.0, step=0.1)
        user_input[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Probability of Fraud: {probability:.2f})")

# -------------------------------
# Model Evaluation
# -------------------------------
elif section == "Model Evaluation":
    st.title("📈 Model Evaluation")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader("ROC AUC Score")
    st.metric("ROC AUC", round(roc_auc_score(y_test, y_proba), 4))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label="ROC Curve")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

# -------------------------------
# Conclusion
# -------------------------------
elif section == "Conclusion":
    st.title("📌 Project Conclusion")
    st.markdown("""
    - Fraud detection is a rare but important classification problem.
    - XGBoost performed well with a strong ROC AUC score.
    - Live prediction allows real-time transaction analysis.

    ### 🔮 Next Steps
    - Add SMOTE or under-sampling
    - Include SHAP explanations
    - Deploy as API service or integrate with transaction pipeline
    """)
