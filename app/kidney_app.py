import streamlit as st
import pickle
import numpy as np

def display():
    with st.form('Kidney_disease_prediction'):
        # Load the scaler and model
        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\kidney_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\kidney_knn.pkl', 'rb') as f:
            model = pickle.load(f)

        # Streamlit app title and description
        st.title("Kidney Disease Prediction")
        st.write("Enter the test values below to predict the likelihood of kidney disease.")

        # Input fields for each feature
        blood_pressure = st.number_input("Blood Pressure", min_value=0, format="%d")
        specific_gravity = st.number_input("Specific Gravity", min_value=0.0, max_value=2.0, format="%.2f")
        albumin = st.number_input("Albumin", min_value=0, format="%d")
        blood_sugar = st.number_input("Blood Sugar", min_value=0, format="%d")
        blood_urea = st.number_input("Blood Urea", min_value=0, format="%d")
        hemoglobin = st.number_input("Hemoglobin", min_value=0.0, format="%.2f")
        white_blood_cells = st.number_input("White Blood Cells", min_value=0, format="%d")
        red_blood_cells = st.number_input("Red Blood Cells", min_value=0, format="%d")

        # Predict button
        if st.form_submit_button("Predict"):
            # Prepare feature array
            features = np.array([[blood_pressure, specific_gravity, albumin, blood_sugar,
                                  blood_urea, hemoglobin, white_blood_cells, red_blood_cells]])

            # Scale the input features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)

            # Display result
            result = "Positive for Kidney Disease" if prediction[0] == 1 else "Negative for Kidney Disease"
            st.write("Prediction:", result)

if __name__ == '__main__':
    display()
