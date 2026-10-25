# Use of ollama using REST API
import requests
import json

# 1. The endpoint for chat
url = "http://localhost:11434/api/chat"

# 2. The payload matches Ollama's expected format
payload = {
    "model": "ministral-3:8b",  # Ensure you've run 'ollama pull ministral-3:8b' to have the model locally
    "messages": [
        {"role": "user", "content": "Why is the sky blue?"}
    ],
    "stream": False  # Set to True for streaming responses
}

# 3. Send the POST request
response = requests.post(url, json=payload)

# 4. Check for success and print the answer
if response.status_code == 200:
    result = response.json()
    print(result["message"]["content"])
else:
    print(f"Error: {response.status_code} - {response.text}")