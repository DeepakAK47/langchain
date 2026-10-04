# building simple llm based application which take input generate 5line summary of topic
from email import parser
import os
import string

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

# define the input for the prompt template
input_data = {"topic": "Virat Kohli"}

# create a chain using the prompt template and the model    
chain = template | model | parser

# invoke the chain with the input data
result = chain.invoke(input_data)
print(result)