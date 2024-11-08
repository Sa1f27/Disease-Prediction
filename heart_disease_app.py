import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

# Load the scaler
with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\scaler.pkl', 'rb') as scaler_pickle:
    scaler = pickle.load(scaler_pickle)

# Load the KNN model
with open(r'C:\Users\huzai\vs code projects\Diseases\predictors\Heart_Disease_KNN.pkl', 'rb') as knn_file:
    knn_model = pickle.load(knn_file)

# Load the Neural Network model
with open(r"C:\Users\huzai\vs code projects\Diseases\models\HeartDisease.json", "r") as json_file:
    loaded_model_json = json_file.read()
neural_model = model_from_json(loaded_model_json)
neural_model.load_weights(r"C:\Users\huzai\vs code projects\Diseases\models\HeartDisease.h5")

st.title("Heart Disease Prediction App")

# Define input fields
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

# Mapping input values to the model's encoding
sex = 0 if sex == "Male" else 1
chest_pain_type = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain_type]
fasting_bs = 1 if fasting_bs == "Yes" else 0
resting_ecg = {"Normal": 0, "ST": 1, "LHV": 2}[resting_ecg]
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope = {"Flat": 0, "Up": 1, "Down": 2}[st_slope]


# Prepare the input data for prediction
input_data = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
                        resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

# Load the scaler used in training
scaled_data = scaler.transform(input_data)

knn = knn_model.predict(scaled_data)
nn = neural_model.predict(scaled_data)

print (knn)
print(nn)
# Prediction button
if st.button("Predict"):
    # KNN prediction
    knn_prediction = knn_model.predict(scaled_data)
    knn_result = "Heart Disease" if knn_prediction[0] == 1 else "No Heart Disease"
    
    # Neural Network prediction
    nn_prediction = neural_model.predict(scaled_data)
    nn_result = "Heart Disease" if nn_prediction[0][0] > 0.5 else "No Heart Disease"

    # Display the results
    st.write(f"KNN Model Prediction: {knn_result}")
    st.write(f"Neural Network Model Prediction: {nn_result}")
