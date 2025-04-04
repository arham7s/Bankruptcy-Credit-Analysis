# -*- coding: utf-8 -*-
"""Bankruptcy-Credit-Analysis.ipynb
# bankruptcy_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --------------------- Utility Functions ---------------------

def train_logistic_model(X, y):
    X = X.apply(pd.to_numeric, errors='coerce').interpolate()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X

def predict_bankruptcy_prob(model, X, firm_names=None):
    X = X.apply(pd.to_numeric, errors='coerce').interpolate()
    prob = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "Firm": firm_names if firm_names is not None else range(len(prob)),
        "Bankruptcy Probability (%)": (prob * 100).round(2)
    }).sort_values(by="Bankruptcy Probability (%)", ascending=False)

def compute_altman_z_score(X1, X2, X3, X4, X5):
    Z = 1.2 * pd.to_numeric(X1, errors='coerce') + \
        1.4 * pd.to_numeric(X2, errors='coerce') + \
        3.3 * pd.to_numeric(X3, errors='coerce') + \
        0.6 * pd.to_numeric(X4, errors='coerce') + \
        1.0 * pd.to_numeric(X5, errors='coerce')
    return Z

def convert_z_to_rating(z):
    if z > 3.0:
        return "AAA"
    elif z > 2.0:
        return "BBB"
    elif z > 1.0:
        return "CCC"
    else:
        return "Default"

def train_credit_rating_model(X, y_ratings):
    X = X.apply(pd.to_numeric, errors='coerce').interpolate()
    X_train, X_test, y_train, y_test = train_test_split(X, y_ratings, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def predict_credit_rating(model, X, firm_names=None):
    X = X.apply(pd.to_numeric, errors='coerce').interpolate()
    preds = model.predict(X)
    return pd.DataFrame({
        "Firm": firm_names if firm_names is not None else range(len(preds)),
        "Predicted Credit Rating": preds
    })

# --------------------- Streamlit UI ---------------------

st.set_page_config(page_title="Bankruptcy Credit Analysis Dashboard", layout="wide")
st.title("\ud83d\udcb8 Bankruptcy Risk & Credit Rating Analysis")

uploaded_file = st.file_uploader("Upload your financial dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    if 'Bankrupt' not in df.columns:
        st.error("Column 'Bankrupt' not found in dataset. Required for model training.")
    else:
        # Extract data
        X = df.drop(columns=['Bankrupt', 'Id'], errors='ignore')
        y = df['Bankrupt']
        firm_names = df['Id'] if 'Id' in df.columns else None

        # Bankruptcy Probability Model
        model, X_full = train_logistic_model(X, y)
        prob_df = predict_bankruptcy_prob(model, X_full, firm_names)

        # Altman Z-Score
        try:
            Z = compute_altman_z_score(df['X1'], df['X2'], df['X3'], df['X4'], df['X5'])
            ratings = Z.map(convert_z_to_rating)
            altman_df = pd.DataFrame({
                "Firm": firm_names if firm_names is not None else range(len(Z)),
                "Altman Z-Score": Z.round(2),
                "Altman Zone": ratings
            })
        except Exception as e:
            st.error(f"Error computing Altman Z-score: {e}")
            altman_df = None

        # Credit Rating Classifier
        if altman_df is not None:
            clf = train_credit_rating_model(X, ratings)
            rating_df = predict_credit_rating(clf, X, firm_names)
        else:
            rating_df = None

        # Merge Outputs
        final = prob_df.set_index("Firm")
        if altman_df is not None:
            final = final.join(altman_df.set_index("Firm"))
        if rating_df is not None:
            final = final.join(rating_df.set_index("Firm"))

        st.subheader("\ud83d\udd22 Risk Scores and Ratings")
        st.dataframe(final.reset_index().sort_values(by="Bankruptcy Probability (%)", ascending=False))

