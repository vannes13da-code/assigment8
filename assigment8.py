
import streamlit as st
import pandas as pd
import joblib

st.title("Missing Persons Status Classification")

st.write(
    "Enter the missing person's information to estimate "
)

# Load the trained model
artifacts = joblib.load("missing_persons_random_forest.pkl")
model = artifacts["pipeline"]

# Load categorical options
options = joblib.load("missing_persons_input_options.pkl")

# Input fields
age = st.number_input(
    "Age", min_value=0, max_value=120, value=25
)

gender = st.selectbox("Gender", options["sexo"])

zone = st.selectbox("Zone", options["zona"])

nationality = st.selectbox(
    "Nationality", options["nacionalidad"]
)

province = st.selectbox(
    "Province", options["provincia"]
)

canton = st.selectbox("Canton", options["canton"])

ethnicity = st.selectbox(
    "Ethnicity", options["etnia"]
)

# Predict
if st.button("Predict", key="predict_status"):

    input_data = pd.DataFrame({
        "edad": [age],
        "sexo": [gender],
        "zona": [zone],
        "nacionalidad": [nationality],
        "provincia": [province],
        "canton": [canton],
        "etnia": [ethnicity]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Status: {prediction[0]}"
    )

    st.write(
        "This is an experimental classification based "
        "on historical records, not a reliable prediction "
        "of an individual's actual outcome."
    )
