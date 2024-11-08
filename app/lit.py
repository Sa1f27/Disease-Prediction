import streamlit as st
import pickle
import numpy as np

model_path = '\home.py'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit UI setup
st.title("Disease Prediction App")
st.sidebar.header("Disease Selection")
st.write("This app predicts if you have a particular disease based on your health data. Currently supports Heart Disease.")

# Main content with tabs for multiple diseases
tabs = st.tabs(["Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke", "Alzheimer's"])

with tabs[0]:  # Heart Disease tab
    st.header("Heart Disease Prediction")

    # Input form for Heart Disease
    with st.form("heart_disease_form"):
        Age = st.number_input("Age", min_value=1, max_value=120, value=25)
        Sex = st.selectbox("Sex", ("Male", "Female"))
        ChestPainType = st.selectbox("Chest Pain Type", ("ATA", "NAP", "ASY", "TA"))
        RestingBP = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
        Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
        RestingECG = st.selectbox("Resting ECG Results", ("Normal", "ST", "LHV"))
        MaxHR = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=220, value=150)
        ExerciseAngina = st.selectbox("Exercise-induced Angina", ("Yes", "No"))
        Oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
        ST_Slope = st.selectbox("ST Slope", ("Flat", "Up", "Down"))

        # Process inputs for model
        Sex = 0 if Sex == "Male" else 1
        ChestPainType = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[ChestPainType]
        FastingBS = 1 if FastingBS == "Yes" else 0
        RestingECG = {"Normal": 0, "ST": 1, "LHV": 2}[RestingECG]
        ExerciseAngina = 1 if ExerciseAngina == "Yes" else 0
        ST_Slope = {"Flat": 0, "Up": 1, "Down": 2}[ST_Slope]

        # Create input array
        input_data = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                                RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])

        # Prediction
        submit_button = st.form_submit_button("Predict")
        if submit_button:
            prediction = model.predict(input_data)
            result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
            st.success(f"Prediction Result: {result}")

# Additional tabs for other diseases
for tab_name in tabs[1:]:
    with tab_name:
        st.header(f"{tab_name} Prediction")
        st.write("Coming soon. Please check back later.")
