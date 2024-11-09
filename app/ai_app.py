import streamlit as st
import pickle
import google.generativeai as genai
import PyPDF2
import numpy as np
#pip install streamlit google-generativeai PyPDF2

# Configure the Gemini API key
genai.configure(api_key="YOUR_API_KEY")  # Replace with your actual API key

# Define a function to read and extract text from PDF
def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"An error occurred while reading the PDF: {e}")
        return None

# Define the Streamlit app
def display():
    st.title("Disease Prediction and Expert Advice")

    # Upload and parse PDF file
    uploaded_file = st.file_uploader("Upload a PDF file with medical details", type="pdf")
    extracted_text = ""
    if uploaded_file:
        extracted_text = read_pdf(uploaded_file)
        if extracted_text:
            st.write("**Extracted Text from PDF:**")
            st.text(extracted_text)
    
    # Define form for disease prediction and advice generation
    with st.form("disease_prediction_form"):
        if extracted_text:
            prompt = f"""Based on the following medical details, provide the best advice and a possible diagnosis:
            {extracted_text}
            Please analyze and suggest potential next steps for managing the condition, considering a range of possible diseases."""

            # Initialize Gemini model
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                st.write("**AI-Generated Response:**")
                st.write(response.text if response else "No response generated. Check your input.")
            except Exception as e:
                st.error(f"An error occurred during AI response generation: {e}")

        # Submit button
        if st.form_submit_button("Generate Prediction and Advice") and not extracted_text:
            st.warning("Please upload a PDF file to analyze.")

if __name__ == "__main__":
    display()
