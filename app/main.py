import streamlit as st
from heart_app import display as heart_disease_display
from kidney_app import display as kidney_disease_display
from diabetes_app import display as diabetes_display
from liver_app import display as liver_disease_display
from stroke_app import display as stroke_display
from ai_app import display as ai_display
# from alzheimers_app import display as alzheimers_display

tabs = st.tabs(["Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke", "Disease Detection"])

with tabs[0]:  # Heart Disease tab
    heart_disease_display()

with tabs[1]:  # Kidney Disease tab
    kidney_disease_display()

with tabs[2]:  # Diabetes tab
    diabetes_display()

with tabs[3]:  # Liver Disease tab
    liver_disease_display()

with tabs[4]:  # Stroke tab
    stroke_display()

with tabs[5]:  # AI's tab
    ai_display()

