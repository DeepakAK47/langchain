# Use of ollama using langchain wrapper
from langchain_ollama import ChatOllama

# Initialize the model (make sure 'ministral-3:8b' is pulled)
model = ChatOllama(
    model="ministral-3:8b",
    temperature=0.7
)

# Invoke
response = model.invoke("What is the capital of India?")
print(response.content)