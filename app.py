import streamlit as st
import numpy as np
import joblib

# Chargement des meilleurs modèles
scaler = joblib.load("scaler.joblib")
best_reg_model = joblib.load("arbre_regression.joblib")  # Meilleur modèle de régression
best_clf_model = joblib.load("svm_classification.joblib")  # Meilleur modèle de classification

# Interface utilisateur
st.title("Prédiction de prix de maison - Boston Housing")

# Choix du modèle
model_type = st.selectbox("Type de modèle", ["Régression", "Classification"])

# Entrée des caractéristiques
features = []
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
for col in columns:
    value = st.number_input(f"{col}", value=0.0)
    features.append(value)

features = np.array(features).reshape(1, -1)
features = scaler.transform(features)

# Bouton de prédiction
if st.button("Faire une prédiction"):
    if model_type == "Régression":
        pred = best_reg_model.predict(features)
        st.subheader(f"Prix estimé : {pred[0]:.2f} $")
    else:
        pred = best_clf_model.predict(features)
        st.subheader("Prix estimé : ÉLEVÉ" if pred[0] == 1 else "Prix estimé : FAIBLE")
