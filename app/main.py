import streamlit as st
from heart_app import display as heart_disease_display
from kidney_app import display as kidney_disease_display
from diabetes_app import display as diabetes_display
from liver_app import display as liver_disease_display
from stroke_app import display as stroke_display
from ai_app import display as ai_display
from ai_app import queries as ai_queries

# Set the page configuration
st.set_page_config(page_title="Disease Diagnostic App", layout="wide")

# Header for the homepage
st.title("Welcome to Disease Diagnostic App")
st.write("""
    A comprehensive platform to predict various diseases such as Heart Disease, Kidney Disease, Diabetes, 
    Liver Disease, Stroke, and AI-powered queries. Our app is designed to assist medical professionals and 
    individuals in quickly assessing the likelihood of certain diseases based on user inputs.
    
    Explore each section for a detailed analysis:
    - **Heart Disease**: Predict the likelihood of heart disease based on user data.
    - **Kidney Disease**: Assess the chances of kidney disease.
    - **Diabetes**: Evaluate the probability of diabetes onset.
    - **Liver Disease**: Predict liver disease conditions based on medical test results.
    - **Stroke**: Check the risk of stroke based on user information.
    - **AI Assistance**: Leverage the power of AI to help with medical predictions and advice.

    Enjoy our easy-to-use interface and feel free to explore each of the disease diagnostic modules.
""")

# Add a tab layout for disease prediction sections
tabs = st.tabs(["AI Assistance", "Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke"])

with tabs[0]:  # AI's tab
    ai_display()
    ai_queries()

with tabs[1]:  # Heart Disease tab
    heart_disease_display()

with tabs[2]:  # Kidney Disease tab
    kidney_disease_display()

with tabs[3]:  # Diabetes tab
    diabetes_display()

with tabs[4]:  # Liver Disease tab
    liver_disease_display()

with tabs[5]:  # Stroke tab
    stroke_display()

# Footer with team information
st.write("""
    --- 

    ### Meet the Team
    Our app was developed by a passionate team of healthcare technology enthusiasts. 
    We aim to make disease diagnosis more accessible and help in early detection of conditions.

    - **Team Member 1**: [LinkedIn](https://www.linkedin.com/in/member1)
    - **Team Member 2**: [LinkedIn](https://www.linkedin.com/in/member2)
    - **Team Member 3**: [LinkedIn](https://www.linkedin.com/in/member3)

    Feel free to reach out to us for more information or inquiries!
""")
