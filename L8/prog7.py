# Runnable Lambda
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# initialising gemini model

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))
parser = StrOutputParser()

# Step 2: Prompt
template = PromptTemplate(
    input_variables=["topic"],
    template="Write a 150-word blog post about {topic}."
)

# Step 3: Custom function 1 — count words
def count_words(text: str) -> dict:
    word_count = len(text.split())
    return {"text": text, "word_count": word_count}

# Step 4: Custom function 2 — estimate reading time
def add_reading_time(data: dict) -> dict:
    words = data["word_count"]
    minutes = round(words / 200, 2)  # Avg reading speed: 200 wpm
    return {
        "text": data["text"],
        "word_count": words,
        "reading_time_minutes": minutes
    }

# Step 5: Custom function 3 — format final output
def format_output(data: dict) -> str:
    return (
        f"--- BLOG POST ---\n{data['text']}\n\n"
        f"--- STATS ---\n"
        f"Word Count: {data['word_count']}\n"
        f"Reading Time: {data['reading_time_minutes']} minutes"
    )

# Step 6: Build chain with RunnableLambda
chain = (
    template
    | model
    | parser
    | RunnableLambda(count_words)
    | RunnableLambda(add_reading_time)
    | RunnableLambda(format_output)
)

# Step 7: Invoke
result = chain.invoke({"topic": "The importance of sleep"})
print(result)
