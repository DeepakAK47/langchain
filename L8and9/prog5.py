# RunnableParrallel
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# initialising gemini model

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))
parser = StrOutputParser()

# Step 3: Twitter prompt
twitter_template = PromptTemplate(
    input_variables=["topic"],
    template="""Write a Twitter post about {topic}.
Rules:
- Max 280 characters
- Include 2-3 relevant hashtags
- Make it punchy and engaging
"""
)

# Step 4: LinkedIn prompt
linkedin_template = PromptTemplate(
    input_variables=["topic"],
    template="""Write a LinkedIn post about {topic}.
Rules:
- Professional and insightful tone
- 3-4 short paragraphs
- End with a thought-provoking question
"""
)

# Step 5: Individual chains
twitter_chain = twitter_template | model | parser
linkedin_chain = linkedin_template | model | parser

# Step 6: Parallel chain
parallel_chain = RunnableParallel(
    twitter=twitter_chain,
    linkedin=linkedin_chain
)

# Step 7: Invoke with topic
result = parallel_chain.invoke({"topic": "The future of AI in healthcare"})

# Step 8: Display results
print("=" * 60)
print("TWITTER POST:")
print("=" * 60)
print(result["twitter"])

print("\n" + "=" * 60)
print("LINKEDIN POST:")
print("=" * 60)
print(result["linkedin"])