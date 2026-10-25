# USe of ollama using wrapper library
from ollama import Client

# Initialize the client pointing to your local server
client = Client(host='http://localhost:11434')

# Send a chat request
response = client.chat(
    model='ministral-3:8b',
    messages=[{'role': 'user', 'content': 'What is the capital of India?'}]
)

print(response['message']['content'])