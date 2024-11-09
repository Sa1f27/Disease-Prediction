import streamlit as st
import pickle
import numpy as np

def display():
    with st.form("Heart_disease_form"): 
        # Load the scaler and model
        try:
            with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\heart_scaler.pkl", 'rb') as scaler_pickle:
                scaler = pickle.load(scaler_pickle)

            with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\heart_knn.pkl", 'rb') as knn_file:
                knn_model = pickle.load(knn_file)
        except FileNotFoundError:
            st.error("Model files not found. Please check the file paths.")
            return

        # Streamlit app title and description
        st.title("Heart Disease Prediction")
        st.write("Enter the details below to predict the likelihood of heart disease.")

        # Input fields for Heart Disease prediction
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        chest_pain_type = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=300)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
        resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LHV"])
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=300)
        exercise_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1)
        st_slope = st.selectbox("ST Slope", options=["Flat", "Up", "Down"])

        # Map categorical inputs to numerical values
        sex = 0 if sex == "Male" else 1
        chest_pain_type = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain_type]
        fasting_bs = 1 if fasting_bs == "Yes" else 0
        resting_ecg = {"Normal": 0, "ST": 1, "LHV": 2}[resting_ecg]
        exercise_angina = 1 if exercise_angina == "Yes" else 0
        st_slope = {"Flat": 0, "Up": 1, "Down": 2}[st_slope]

        # Prepare the input data for prediction
        input_data = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
                                resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Prediction button
        if st.form_submit_button("Predict"):
            # KNN prediction
            knn_prediction = knn_model.predict(scaled_data)
            knn_result = "Heart Disease" if knn_prediction[0] == 1 else "No Heart Disease"

            # Display the results
            st.write(f"KNN Model Prediction: {knn_result}")

if __name__ == '__main__':
    display()
