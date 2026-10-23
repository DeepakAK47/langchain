# building agent using modern agent code
import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

# 1. Initialize your model (exactly as you requested)
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 2. Define a simple, reliable tool (no external search API needed)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Using a free, stable weather API that doesn't require a key for basic use
    try:
        # wttr.in returns plain text weather, no API key required
        response = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        if response.status_code == 200:
            return response.text
        return f"Could not fetch weather for {city}."
    except Exception as e:
        return f"Error: {str(e)}"


# 3. Create the agent using the modern API
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use the get_weather tool to answer weather questions."
)

# 4. Invoke the agent
response = agent.invoke({
    "messages": [("user", "What is the weather like in Delhi right now?")]
})

# 5. Extract the text from the response blocks (as you learned earlier)
final_message = response["messages"][-1]
if isinstance(final_message.content, list):
    print(final_message.content[0].get("text", ""))
else:
    print(final_message.content)