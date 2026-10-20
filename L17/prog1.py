# call binding, tool calling, tool execution
from json import tool
import os
from urllib import response
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()   

@tool
def add_num(a:int,b:int) -> int:
    """Add two numbers"""
    return a + b

@tool
def weather_data(city: str) -> dict:
    """Get weather data for a city"""
    # This is a placeholder implementation
    return {"city": city, "temperature": 25, "description": "Sunny"}


model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

tools = [add_num, weather_data]
model_with_tools = model.bind_tools(tools)  # adding tools to the model

response = model_with_tools.invoke("Add 5 and 10, then get the weather for New York City.")

print(response)