#  Runnable branching with RunnableBranch
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

# initialising gemini model

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))
parser = StrOutputParser()

# ---------- Define three specialized chains ----------

refund_template = PromptTemplate(
    input_variables=["query"],
    template="You are a refund specialist. Handle this refund request politely:\n{query}"
)

tech_template = PromptTemplate(
    input_variables=["query"],
    template="You are a technical support engineer. Diagnose and solve:\n{query}"
)

general_template = PromptTemplate(
    input_variables=["query"],
    template="You are a friendly assistant. Answer this:\n{query}"
)

refund_chain = refund_template | model | parser
tech_chain = tech_template | model | parser
general_chain = general_template | model | parser

def is_refund(x):
    return any(word in x["query"].lower() for word in ["refund", "return", "money back"])

def is_technical(x):
    return any(word in x["query"].lower() for word in ["error", "bug", "crash", "not working"])

# ---------- Build RunnableBranch ----------

branch = RunnableBranch(
    (RunnableLambda(is_refund), refund_chain),
    (RunnableLambda(is_technical), tech_chain),
    general_chain   # default
)

# ---------- Test with different queries ----------

queries = [
    "I want a refund for my last order",
    "The app keeps crashing on startup",
    "What are your business hours?"
]

for q in queries:
    print(f"\nQuery: {q}")
    print(f"Response: {branch.invoke({'query': q})}")