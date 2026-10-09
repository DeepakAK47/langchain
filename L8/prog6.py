# use of runnable pass through
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# initialising gemini model

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))
parser = StrOutputParser()

# Twitter prompt
twitter_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a Twitter post about {topic}. Max 280 chars, 2-3 hashtags."
)

# LinkedIn prompt
linkedin_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a LinkedIn post about {topic}. Professional, 3 paragraphs."
)

# Individual chains
twitter_chain = twitter_template | model | parser
linkedin_chain = linkedin_template | model | parser

# Parallel chain with RunnablePassthrough
parallel_chain = RunnableParallel(
    twitter=RunnablePassthrough() | twitter_chain,
    linkedin=RunnablePassthrough() | linkedin_chain
)

# Invoke
result = parallel_chain.invoke({"topic": "The future of AI in healthcare"})

print("TWITTER:\n", result["twitter"])
print("\nLINKEDIN:\n", result["linkedin"])

