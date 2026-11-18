
import streamlit as st
import pandas as pd
import joblib

st.title("Missing Persons Clustering")
st.write("Enter the missing person's information to identify the corresponding cluster.")

# Load the trained model and preprocessor
model = joblib.load("kmeans_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Load the original dataset
data = pd.read_excel(
    "mdi_personasdesaparecidas_pm_2026_enero_julio.xlsx",
    sheet_name="1. Pdesaparecidas"
)

# Get available options from the dataset
def get_options(column):
    return sorted(data[column].dropna().astype(str).unique().tolist())

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)

gender = st.selectbox("Gender", get_options("sexo"))

zone = st.selectbox("Zone", get_options("zona"))

nationality = st.selectbox("Nationality", get_options("nacionalidad"))

province = st.selectbox("Province", get_options("provincia"))

canton = st.selectbox("Canton", get_options("canton"))

ethnicity = st.selectbox("Ethnicity", get_options("etnia"))

motivation = st.selectbox("Motivation", get_options("motivacion"))

observed_motivation = st.selectbox(
    "Observed Motivation",
    get_options("motivacion_observada")
)

# Localization date
not_applicable = st.checkbox("Localization Date: Not Applicable", value=True)

if not_applicable:
    localization_date = "NO_APLICA"
else:
    selected_date = st.date_input("Localization Date")
    localization_date = selected_date.strftime("%Y/%m/%d")

# Predict cluster
if st.button("Predict", key="predict_cluster"):

    input_data = pd.DataFrame({
        "edad": [age],
        "sexo": [gender],
        "zona": [zone],
        "nacionalidad": [nationality],
        "provincia": [province],
        "canton": [canton],
        "etnia": [ethnicity],
        "fecha_localizacion": [localization_date],
        "motivacion": [motivation],
        "motivacion_observada": [observed_motivation]
    })

    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)

    st.write(f"The predicted cluster is: {prediction[0]}")
