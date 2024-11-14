# SoulScan: Advance Disease Diagnosis with  ML & AI

This platform is designed to predict various diseases, including Heart Disease, Kidney Disease, Diabetes, Liver Disease, Stroke, and AI-powered assistance. The app leverages machine learning algorithms and large datasets to analyze user data and provide reliable disease predictions with over **90% accuracy**.

## Features

- **Heart Disease**: Predicts the likelihood of heart disease based on user data.
- **Kidney Disease**: Assesses the chances of kidney disease.
- **Diabetes**: Evaluates the probability of diabetes onset.
- **Liver Disease**: Predicts liver disease conditions based on medical test results.
- **Stroke**: Checks the risk of stroke based on user information.
- **AI Assistance**: Uses AI-powered tools to provide additional medical predictions and advice.
- **Upload Medical Report**: Users can upload their medical reports for analysis and receive personalized health recommendations.

## Installation

To run the app locally, follow these steps:

1. **Clone the repository**:

   ```
   git clone https://github.com/Sa1f27/Disease-Prediction.git
   cd Disease-Prediction
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the app**:

   In the project directory, run:

   ```
   cd app
   streamlit run app.py
   ```

   This will start the app and open it in your browser.

2. **Upload a medical report**:
   - Go to the "Upload Your Medical Report" section in the app.
   - Upload the report (e.g., a PDF, CSV, or image depending on the supported formats).
   - Wait for the analysis to complete, and you will receive suggestions and next steps based on the results.

3. **Explore the disease prediction modules**:
   - Navigate through the app to explore each disease prediction module.
   - Input relevant health information to predict the likelihood of the disease.
     
![Screenshot 2024-11-10 123012](https://github.com/user-attachments/assets/4edc575d-c96c-44da-8110-cf3688e39742)

![Screenshot 2024-11-10 123031](https://github.com/user-attachments/assets/385595e2-8b11-4e99-8f14-70c1a4a8f665)

![Screenshot 2024-11-10 123041](https://github.com/user-attachments/assets/f0b438c2-495c-4451-a091-b2d58215c848)

![Screenshot 2024-11-10 123104](https://github.com/user-attachments/assets/138bb722-bb17-4a2b-a936-e763234bd0d1)

![Screenshot 2024-11-10 123129](https://github.com/user-attachments/assets/c1c118a7-ffa3-4120-9b2d-d93578223b01)

![Screenshot 2024-11-10 123200](https://github.com/user-attachments/assets/136d1f2d-8f06-48bc-8c45-e75bd6e973ae)

![Screenshot 2024-11-10 123213](https://github.com/user-attachments/assets/9bf357fb-1327-4606-a4fa-d7a0022af54d)

## How It Works

- The app uses **machine learning** models trained on large datasets to predict disease risk.
- The algorithms analyze user inputs, such as age, gender, medical test results, and lifestyle factors, to generate predictions.
- The system provides personalized recommendations based on the analysis, which could include further medical tests, lifestyle changes, or other suggestions.

## Technologies Used

- **Streamlit**: For building the web interface.
- **Python**: Programming language for backend logic.
- **Scikit-learn**: For machine learning models.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For data visualization.
