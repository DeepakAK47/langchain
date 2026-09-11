# building chatbot
import os
import streamlit as st
from google import genai
from dotenv import load_dotenv
load_dotenv()

# Initialize the Google GenAI client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# create a chat session state to store the conversation history
chat = client.chats.create(model="gemini-3.6-flash")

#streamlit interface

st.title("Gemini Chatbot")

# Initialize chat history in session state (persists across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Send to Gemini
    with st.chat_message("assistant"):
        response = chat.send_message(prompt)
        st.markdown(response.text)
    
    # Store in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response.text})
