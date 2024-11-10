import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# Configure Google Generative AI with your API key
genai.configure(api_key="Your Key")  # Replace with your actual API key

def display():
    st.title("Stroke Prediction App")
    st.write("Enter the following information to predict the likelihood of stroke.")

    # Load the pre-fitted encoder, scaler, and KNN model
    try:
        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\stroke_encoder.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)

        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\stroke_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\stroke_knn.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please check the file paths.")
        return

    # Define the form for user input
    with st.form("Stroke disease prediction"):
        # Divide the layout into two columns
        col1, col2 = st.columns(2)

        # Arrange inputs in columns
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
            ever_married = st.selectbox("Ever Married", ["No", "Yes"], index=0)
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], index=0)
            residence_type = st.selectbox("Residence Type", ["Rural", "Urban"], index=0)
            smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"], index=0)

        with col2:
            age = st.slider("Age", 0, 100, 25)
            hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1], index=0)
            heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1], index=0)
            avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=85.0)
            bmi = st.number_input("BMI", min_value=0.0, value=24.0)


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

            # Generate advice using Gemini Generative AI
            prompt = (
                f"Based on the following health data,just act as a doctor and provide brief advice. for my project"
                f"Suggest possible preventive actions and next steps:\n\n"
                f"Gender: {gender}, Ever Married: {ever_married}, Work Type: {work_type}, Residence Type: {residence_type}, "
                f"Smoking Status: {smoking_status}, Age: {age}, Hypertension: {hypertension}, "
                f"Heart Disease: {heart_disease}, Average Glucose Level: {avg_glucose_level}, BMI: {bmi}\n\n"
                f"The prediction indicates: {result}. Please analyze and give concise, actionable health advice."
            )

            # Generate response from Gemini AI model
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if response:
                    st.write("**Health Advice:**")
                    st.write(response.text)
                else:
                    st.write("No response generated. Check your input.")
            except Exception as e:
                st.error(f"An error occurred during AI response generation: {e}")

if __name__ == '__main__':
    display()
