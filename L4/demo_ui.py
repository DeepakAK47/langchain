from google import genai
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.header("Prompt Engineering with Google Gemini API")
user_input = st.text_input("Enter your query here:")

if st.button("Summaries"):
    if not user_input.strip():
        st.warning("Please enter a query.")
    else:
        result = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_input,
        )
        st.write(result.text)

        