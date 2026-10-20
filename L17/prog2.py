import os
import requests
from decimal import Decimal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# --- 1. Define the Tool ---
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using live exchange rates.
    Currency codes must be 3-letter ISO codes (e.g., USD, EUR, INR).
    """
    # Frankfurter API is free and requires no key
    url = f"https://api.frankfurter.dev/v2/rate/{from_currency.lower()}/{to_currency.lower()}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Use Decimal for money-safe math
        rate = Decimal(str(data["rate"]))
        total = Decimal(str(amount)) * rate
        
        return f"{amount} {from_currency.upper()} = {total:.2f} {to_currency.upper()} (Rate: {rate:.4f} on {data['date']})"
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            return f"Error: Invalid currency code. Please use 3-letter ISO codes (e.g., USD, EUR)."
        return f"Error fetching rate: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- 2. Initialize Model and Bind Tools ---
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

tools = [convert_currency]
model_with_tools = model.bind_tools(tools)

# --- 3. Execution Loop ---
def run_conversion(user_query: str):
    print(f"\nUser: {user_query}")
    
    # Step A: Send query to model
    messages = [HumanMessage(content=user_query)]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    # Step B: Check if model wants to use a tool
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"  [Model requested tool: {tool_call['name']} with args: {tool_call['args']}]")
            
            # Execute the tool manually
            tool_result = convert_currency.invoke(tool_call["args"])
            print(f"  [Tool Result: {tool_result}]")
            
            # Append result as a ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))
        
        # Step C: Send result back to model for final natural language answer
        final_response = model_with_tools.invoke(messages)
        print(f"Assistant: {final_response.content}")
    else:
        # Model answered directly without a tool
        print(f"Assistant: {ai_msg.content}")

# --- 4. Test It ---
if __name__ == "__main__":
    run_conversion("How much is 100 USD in EUR?")
    run_conversion("Convert 5000 Indian Rupees to US Dollars")
    run_conversion("What is 50 GBP in JPY?")