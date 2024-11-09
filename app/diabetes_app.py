import streamlit as st
import pickle
import numpy as np
from ai_app import display as ai_display
import google.generativeai as genai
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.mixed_precision import policy
from sklearn.preprocessing import StandardScaler

# Configure Google Generative AI
genai.configure(api_key="AIzaSyAe0w7EC0TTrh6tG0Ijd6HGxIFijg_hp50")  # Replace with your actual API key

# Define the main display function
def display():
    st.title("Diabetes Prediction App")

    # Load required resources
    try:
        # Load the scaler
        with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\diabetes_scaler.pkl", 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Load the KNN model
        with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\Diabetes_knn.pkl", 'rb') as knn_file:
            knn_model = pickle.load(knn_file)
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please check the file paths.")
        return

    # Define form for user input
    with st.form("Diabetes_disease_form"):
        # Input fields for user medical details
        # Input fields with default values and sliders for smooth input
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=120, step=1, value=1)
        Glucose = st.slider("Glucose", min_value=0, max_value=200, value=100)
        BloodPressure = st.slider("BloodPressure", min_value=40, max_value=200, value=70)
        SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=300, value=20)
        Insulin = st.number_input("Insulin", min_value=0, max_value=600, value=100)
        BMI = st.slider("BMI", min_value=10, max_value=50, value=25)
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.5)
        Age = st.slider("Age", min_value=0, max_value=120, value=30)


        # Prepare the input data for prediction
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        scaled_data = scaler.transform(input_data)

        # Form submit button
        if st.form_submit_button("Predict"):
            # Make prediction using the KNN model
            knn_prediction = knn_model.predict(scaled_data)
            knn_result = "You have Diabetes" if knn_prediction == 1 else "You don't have Diabetes"
            st.write(f"KNN Model Prediction: {knn_result}")

            # Prepare prompt for Generative AI model
            prompt = (
                f"Based on the following medical details, act as a doctor providing advice for a project. "
                f"Provide the best advice and a possible diagnosis:\n\n"
                f"Pregnancies: {Pregnancies}, Glucose: {Glucose}, Blood Pressure: {BloodPressure}, "
                f"Skin Thickness: {SkinThickness}, Insulin: {Insulin}, BMI: {BMI}, "
                f"Diabetes Pedigree Function: {DiabetesPedigreeFunction}, Age: {Age}\n\n"
                f"I have been diagnosed with {knn_result}. Please analyze and suggest potential next steps "
                f"for managing the condition, and make the response concise and in bullet points."
            )

            # Generate response from Generative AI model
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if response:
                    st.write("**Suggestion:**")
                    st.write(response.text)
                else:
                    st.write("No response generated. Check your input.")
            except Exception as e:
                st.error(f"An error occurred during AI response generation: {e}")

if __name__ == '__main__':
    display()
