import streamlit as st
import pickle
import google.generativeai as genai
import PyPDF2
import numpy as np

# Configure the Gemini API key
genai.configure(api_key="Your Key")  # Replace with your actual API key

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
    st.write("""
    **Upload Your Medical Report**

    To receive a thorough analysis and personalized recommendations, please upload your medical report. Our advanced 
    algorithms will analyze the data and provide you with the best possible suggestions and next steps to take for 
    further diagnosis or treatment. 
             
    - Potential diagnoses
    - Suggested medical tests or screenings to confirm the condition
    - Lifestyle adjustments and preventive measures

    Our goal is to support you in making informed decisions about your health and to assist healthcare professionals 
    in providing the best care possible. Upload your report today, and let us help guide your health journey.""")


    # Upload and parse PDF file
    uploaded_file = st.file_uploader("Upload a PDF file with medical details", type="pdf")
    extracted_text = ""
    if uploaded_file:
        extracted_text = read_pdf(uploaded_file)
        if extracted_text:
            st.write("**PDF Details:**")  # Bold title
            st.text(f"{extracted_text}")  # Bold the extracted text

    
    # Define form for disease prediction and advice generation
    with st.form("disease_prediction_form"):
        if extracted_text:
            prompt = f"""Based on the following medical details,just act as a dummy doctor who is giving advise for my project, provide the best advice and a possible diagnosis:
            {extracted_text}
            Please analyze and suggest potential next steps for managing the condition, considering a range of possible diseases.
            and make the response short and in points"""

            # Initialize Gemini model
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if response:
                    st.session_state.generated_response = response.text  # Save response to session state
                    st.write("**Suggestion:**")
                    st.write(response.text)
                else:
                    st.write("No response generated. Check your input.")
            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")

            # Submit button for disease prediction form
            if st.form_submit_button("Generate Prediction and Advice"):
                st.warning("Please upload a PDF file to analyze.")

def queries():
    # Check if we have a generated response in session state
    if "generated_response" in st.session_state:
        with st.form("query_form"):
            query = st.text_input("Ask your queries:")
            if query:
                prompt = f"{st.session_state.generated_response} I have some more questions and it is: {query}"

                # Initialize Gemini model
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    st.write("**Response:**")
                    st.write(response.text if response else "No response generated. Check your input.")
                except Exception as e:
                    st.error(f"An error occurred during AI response generation: {e}")

            # Submit button for query form
            if st.form_submit_button("Submit Query"):
                if not query:
                    st.warning("Please enter a query.")

    else:
        st.warning("Please generate a disease prediction first by uploading a PDF.")

if __name__ == "__main__":
    display()
    queries()
