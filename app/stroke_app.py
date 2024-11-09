import streamlit as st
import pickle
import numpy as np

def display():
    with st.form("Stroke disease prediction"):
        # Load the pre-fitted encoder, scaler, and model
        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\stroke_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\stroke_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\stroke_knn.pkl', 'rb') as f:
            model = pickle.load(f)

        # App title and user instructions
        st.title("Stroke Prediction App")
        st.write("Enter the following information to predict the likelihood of stroke.")

        # User input fields
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        age = st.slider("Age", 0, 100, 25)
        hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
        heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
        bmi = st.number_input("BMI", min_value=0.0)

        # Prepare input data
        categorical_data = [[gender, ever_married, work_type, residence_type, smoking_status]]
        numerical_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]])

        # Encode categorical data using the pre-fitted encoder
        encoded_data = encoder.transform(categorical_data)

        # Concatenate encoded and numerical data for scaling
        input_data = np.concatenate([encoded_data, numerical_data], axis=1)

        # Scale the combined data
        scaled_data = scaler.transform(input_data)

        # Perform prediction
        if st.form_submit_button("Predict"):
            prediction = model.predict(scaled_data)
            result = "High risk of stroke" if prediction[0] == 1 else "Low risk of stroke"
            st.write("Prediction:", result)

if __name__ == '__main__':
    display()
