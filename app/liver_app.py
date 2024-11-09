import streamlit as st
import pickle
import numpy as np

def display():
    with st.form('Liver_disease_prediction'):
        # Load the scaler and model
        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\liver_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\liver_knn.pkl', 'rb') as f:
            model = pickle.load(f)

        # App title and description
        st.title("Liver Disease Prediction")
        st.write("Enter the medical test values below to predict the likelihood of liver disease.")

        # Input fields for each feature
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, format="%.2f")
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, format="%.2f")
        alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0, format="%d")
        alanine_aminotransferase = st.number_input("Alamine Aminotransferase (Sgpt)", min_value=0, format="%d")
        total_proteins = st.number_input("Total Proteins", min_value=0.0, format="%.2f")
        albumin = st.number_input("Albumin", min_value=0.0, format="%.2f")
        albumin_globulin_ratio = st.number_input("Albumin-Globulin Ratio", min_value=0.0, format="%.2f")

        # Predict button
        if st.form_submit_button("Predict"):
            # Prepare feature array
            features = np.array([[total_bilirubin, direct_bilirubin, alkaline_phosphatase,
                                  alanine_aminotransferase, total_proteins, albumin,
                                  albumin_globulin_ratio]])

            # Scale the input features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)

            # Display result based on the model's output
            result = "Negative for Liver Disease" if prediction[0] == 1 else "Positive for Liver Disease"
            st.write("Prediction:", result)

if __name__ == "__main__":
    display()
