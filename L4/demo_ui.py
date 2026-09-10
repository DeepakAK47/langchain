from google import genai
from google.genai import types
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()


st.header("Prompt Engineering with Google Gemini API")
user_input  = st.text_input("Enter your query here :")

if st.button("Summaries"):
    result = model.invoke()