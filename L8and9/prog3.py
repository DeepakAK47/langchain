# use of LLM chain
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# initialising gemini model

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# prompt template for generating 5 line summary
template = PromptTemplate(
    input_variables=["topic"],
    template="Generate a 5-line summary of the topic: {topic}"
)

parser = StrOutputParser()

# create a chain using the prompt template, model, and output parser
chain = template | model | parser

# run chain with specific topic
input_data = {"topic": "Virat Kohli"}
result = chain.invoke(input_data)
print(result)