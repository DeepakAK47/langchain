# sequential pipelines using chaining
# sequential pipelines using chaining       
import os
import string   

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 1st prompt -> five line summary
template2 = PromptTemplate(
    template='Write a five line summary of {topic}',
    input_variables=['topic']
)

# initialising gemini model

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

parser = StrOutputParser()

# Chain using LCEL
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Virat Kohli'})

print(result) 
chain.get_graph().print_ascii()