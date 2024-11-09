# Install the Google AI Python SDK
#!pip install google-generativeai

# See the getting started guide for more information:
# https://ai.google.dev/gemini-api/docs/get-started/python

# Import the Google AI Python SDK
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyAe0w7EC0TTrh6tG0Ijd6HGxIFijg_hp50")

# Create the model
model = genai.GenerativeModel('gemini-1.5-flash')

# Generate content
prompt = "hello,give your short introduction"
response = model.generate_content(prompt)

# Print the response
print(response.text)