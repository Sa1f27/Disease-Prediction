import streamlit as st

def display():
    with st.form("Diabetes_disease_form"):    
        import pickle
        import numpy as np
        from tensorflow.python.keras.models import model_from_json
        from sklearn.preprocessing import StandardScaler
        from tensorflow.python.keras.models import model_from_json
        from tensorflow.python.keras.mixed_precision import policy

        # Ensure DTypePolicy is recognized as a custom object
        custom_objects = {"DTypePolicy": policy.Policy}

        # Load the scaler
        with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\diabetes_scaler.pkl", 'rb') as scaler_pickle:
            scaler = pickle.load(scaler_pickle)

        # Load the KNN model
        with open(r"C:\Users\huzai\vs code projects\Diseases\predictors\Diabetes_knn.pkl", 'rb') as knn_file:
            knn_model = pickle.load(knn_file)

        # Load the Neural Network model


        st.title(" Diabeties Prediction App")

        # Define input fields
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=120, step=1)
        Glucose = st.number_input("Glucose")
        BloodPressure = st.number_input("BloodPressure")
        SkinThickness= st.number_input("SkinThickness", min_value=0, max_value=300)
        Insulin = st.number_input("Insulin", min_value=0, max_value=600)
        BMI= st.number_input("BMI")
        DiabetesPedigreeFunction= st.number_input("DiabetesPedigreeFunction")
        Age = st.number_input("Age", min_value=0, max_value=120)


        # Mapping input values to the model's encoding



        # Prepare the input data for prediction
        input_data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        # Load the scaler used in training
        scaled_data = scaler.transform(input_data)

        knn = knn_model.predict(scaled_data)


        print (knn)

        # Prediction button
        if st.form_submit_button("Predict"):
            # KNN prediction
            knn_prediction = knn_model.predict(scaled_data)
            knn_result = " Disease Detected" if knn_prediction == 1 else "No Disease detected"
            
            # Neural Network prediction
            

            # Display the results
            st.write(f"KNN Model Prediction: {knn_result}")
            
if __name__=='__main__':
    display()