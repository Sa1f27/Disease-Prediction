import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# Configure Google Generative AI with your API key
genai.configure(api_key="AIzaSyAe0w7EC0TTrh6tG0Ijd6HGxIFijg_hp50")  # Replace with your actual API key

# Define the main display function for the Heart Disease Prediction App
def display():
    st.title("Heart Disease Prediction App")
    st.write("Enter the details below to predict the likelihood of heart disease.")

    # Load the scaler and KNN model
    try:
        with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\heart_scaler.pkl", 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\heart_knn.pkl", 'rb') as knn_file:
            knn_model = pickle.load(knn_file)
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please check the file paths.")
        return

    # Define form for user input
    with st.form("Heart_disease_form"):

        col1, col2 = st.columns(2)

        # Arrange inputs in columns
        with col1:
        # Input fields for heart disease prediction
            age = st.slider("Age", min_value=0, max_value=120, step=1, value=50)
            sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
            chest_pain_type = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"], index=0)
            resting_bp = st.slider("Resting Blood Pressure", min_value=0, max_value=300, value=120, step=1)
            cholesterol = st.slider("Cholesterol", min_value=0, max_value=600, value=200, step=5)
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"], index=1)
            resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LHV"], index=0)
            max_hr = st.slider("Max Heart Rate Achieved", min_value=0, max_value=300, value=150, step=5)
            exercise_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"], index=1)
            oldpeak = st.slider("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST Slope", options=["Flat", "Up", "Down"], index=0)


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
            st.write(f"KNN Model Prediction: {knn_result}")

            # Prepare prompt for Generative AI model
            prompt = (
                f"Based on the following medical details, act as a doctor providing advice for a project. "
                f"Provide the best advice and a possible diagnosis:\n\n"
                f"Age: {age}, Sex: {'Male' if sex == 0 else 'Female'}, Chest Pain Type: {chest_pain_type}, "
                f"Resting Blood Pressure: {resting_bp}, Cholesterol: {cholesterol}, "
                f"Fasting Blood Sugar > 120 mg/dl: {'Yes' if fasting_bs == 1 else 'No'}, "
                f"Resting ECG: {resting_ecg}, Max Heart Rate Achieved: {max_hr}, "
                f"Exercise Induced Angina: {'Yes' if exercise_angina == 1 else 'No'}, Oldpeak: {oldpeak}, "
                f"ST Slope: {st_slope}\n\n"
                f"I have been diagnosed with {knn_result}. Please analyze and suggest potential next steps "
                f"for managing the condition, and make the response concise and in bullet points."
            )

            # Generate response from Generative AI model
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
