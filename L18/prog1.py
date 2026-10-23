# building simple ai agent that fetch data from the browser
import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

load_dotenv()

# initialising gemini model
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# define tool for searching the web using DuckDuckGo
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather data for a given city."""
    url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'
    response = requests.get(url)
    if response.status_code == 200:
        return str(response.json())
    return "Error fetching weather data"


# 3. Create the agent using the modern API
agent = create_agent(
    model=model,
    # tools=[search_tool, get_weather_data],
    system_prompt="You are a helpful assistant. Use the provided tools to answer the user's question."
)

# 4. Invoke the agent
response = agent.invoke({
    "messages": [("user", "Find the capital of Madhya Pradesh, then find its current weather condition")]
})

# 5. Print the final answer
final_message = response["messages"][-1]
print(final_message.content[0]["text"])