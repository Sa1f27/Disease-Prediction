import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# Configure Google Generative AI with your API key
genai.configure(api_key="Your Key")  # Replace with your actual API key

def display():
    st.title("Liver Disease Prediction App")
    st.write("Enter the medical test values below to predict the likelihood of liver disease.")

    # Load the scaler and KNN model
    try:
        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\liver_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\liver_knn.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please check the file paths.")
        return

    # Define form for user input
    with st.form('Liver_disease_prediction'):

        col1, col2 = st.columns(2)

        # Arrange inputs in columns
        with col1:
        # Input fields with default values
            total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=10.0, format="%.2f", value=1.0)
            direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=5.0, format="%.2f", value=0.3)
            alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0, max_value=2000, format="%d", value=100)
        with col2:
            alanine_aminotransferase = st.number_input("Alamine Aminotransferase (Sgpt)", min_value=0, max_value=1000, format="%d", value=20)
            total_proteins = st.number_input("Total Proteins", min_value=0.0, max_value=10.0, format="%.2f", value=6.8)
            albumin = st.number_input("Albumin", min_value=0.0, max_value=5.0, format="%.2f", value=3.5)
            albumin_globulin_ratio = st.number_input("Albumin-Globulin Ratio", min_value=0.0, max_value=5.0, format="%.2f", value=1.1)

        # Prediction button
        if st.form_submit_button("Predict"):
            # Prepare feature array for prediction
            features = np.array([[total_bilirubin, direct_bilirubin, alkaline_phosphatase,
                                  alanine_aminotransferase, total_proteins, albumin,
                                  albumin_globulin_ratio]])

            # Scale the input features
            features_scaled = scaler.transform(features)

            # Make prediction using KNN model
            prediction = model.predict(features_scaled)
            result = "Positive for Liver Disease" if prediction[0] == 1 else "Negative for Liver Disease"
            st.write("KNN Model Prediction:", result)

            # Generate advice using Gemini Generative AI
            prompt = (
                f"Based on the following liver function test results,just act as a doctor and provide brief advice. for my project"
                f"Suggest potential next steps:\n\n"
                f"Total Bilirubin: {total_bilirubin}, Direct Bilirubin: {direct_bilirubin}, Alkaline Phosphatase: {alkaline_phosphatase}, "
                f"Alamine Aminotransferase (Sgpt): {alanine_aminotransferase}, Total Proteins: {total_proteins}, "
                f"Albumin: {albumin}, Albumin-Globulin Ratio: {albumin_globulin_ratio}\n\n"
                f"The patient is diagnosed as {result}. Please analyze and provide short, actionable points for managing the condition."
            )

            # Generate response from Gemini AI model
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if response:
                    st.write("**Medical Advice:**")
                    st.write(response.text)
                else:
                    st.write("No response generated. Check your input.")
            except Exception as e:
                st.error(f"An error occurred during AI response generation: {e}")

if __name__ == '__main__':
    display()
