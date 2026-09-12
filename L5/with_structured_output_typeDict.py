# syntax with proper reason
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
import json

load_dotenv()

# Schema

class schemas(TypedDict):
    summary: str
    key_themes: List[str]
    sentiment: str
    action_items: Optional[List[str]]


# initialize the Google GenAI client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Step 3: Your input text
text_data = """
I recently upgraded to the Samsung Galaxy S24 Ultra. The processor is lightning fast, 
and the 200MP camera is stunning in low light. However, the phone is heavy and expensive.
The battery easily lasts a full day, and the S-Pen is useful for notes.
"""

# Step 4: Call the model with schema
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=f"Extract key insights from this review:\n\n{text_data}",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_themes": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "sentiment": {"type": "string"},
                "action_items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["summary", "key_themes", "sentiment"]
        }
    )
)

# Step 5: Parse the JSON response
data = json.loads(response.text)
print(f"Summary: {data['summary']}")
print(f"Key Themes: {data['key_themes']}")
print(f"Sentiment: {data['sentiment']}")
print(f"Action Items: {data.get('action_items', [])}")