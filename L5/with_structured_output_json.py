from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv
load_dotenv()

# schema
json_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_themes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "action_items": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["summary", "key_themes", "sentiment"]
}

# initialize the Google GenAI client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Step 3: Your input text
text_data = """
I recently upgraded to the Samsung Galaxy S24 Ultra. The processor is lightning fast,
and the 200MP camera is stunning in low light. However, the phone is heavy and expensive.
The battery easily lasts a full day, and the S-Pen is useful for notes.
"""

# call with json schema

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=text_data,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=json_schema
    )
)

# Step 5: Parse the JSON response
data = json.loads(response.text)
print(f"Summary: {data['summary']}")

