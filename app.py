import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("income_model.pkl")

# Titre principal
st.set_page_config(page_title="Prédiction de revenu", page_icon="💰")
st.title("💰 Prédire si une personne gagne plus de 50K$/an")
st.markdown("Entrez les informations socio-professionnelles ci-dessous pour prédire le revenu.")

# Entrées utilisateur
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Âge", 18, 80, 30)
    education_num = st.slider("Niveau d'éducation (1=Low, 16=High)", 1, 16, 9)
    capital_gain = st.number_input("Gain en capital ($)", 0, 100000, 0)
    
with col2:
    hours_per_week = st.slider("Heures travaillées par semaine", 1, 99, 40)
    capital_loss = st.number_input("Perte en capital ($)", 0, 5000, 0)
    sex = st.radio("Sexe", ["Male", "Female"])
    sex_encoded = 1 if sex == "Male" else 0

# Organiser les features
X_input = pd.DataFrame([[
    sex_encoded, age, education_num, capital_gain, capital_loss, hours_per_week
]], columns=["sex", "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"])

# Affichage
st.markdown("---")
if st.button("🔍 Prédire le revenu"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    result = ">50K" if prediction == 1 else "<=50K"
    emoji = "🟢" if prediction == 1 else "🔵"

    st.subheader(f"{emoji} Revenu prédit : **{result}**")
    st.progress(proba)
    st.caption(f"Probabilité d'avoir un revenu >50K : **{proba:.2%}**")

    st.markdown("---")
    st.write("**Données utilisées :**")
    st.dataframe(X_input)
