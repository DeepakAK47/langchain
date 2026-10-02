# json outparser using chain
# take to input and print their output using chain
import os
import string

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()

# initialising gemini model
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# create a json output parser
parser = JsonOutputParser()

# Create prompt with format instructions
prompt = PromptTemplate(
    template="Extract person information as JSON with keys: name, age, occupation\n{format_instructions}\n{text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Build chain
chain = prompt | model | parser

# Invoke
result = chain.invoke({"text": "John Doe is a 35 year old software engineer at Acme Corp"})

print(result)           # {'name': 'John Doe', 'age': 35, 'occupation': 'Software Engineer'}
print(type(result))     # <class 'dict'>