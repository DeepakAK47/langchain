# structure parser using chain
import os
import string

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

# initialising gemini model
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Step 1: Define Pydantic schema
class PersonInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job title or profession")
    city: str = Field(description="City of residence")

# Step 3: Bind structured output
structured_model = model.with_structured_output(PersonInfo)

# Step 4: Create prompt
prompt = PromptTemplate(
    template="Extract person information from this text:\n{text}",
    input_variables=["text"]
)

# Step 5: Build chain
chain = prompt | structured_model

# Step 6: Invoke
result = chain.invoke({
    "text": "John Doe is a 35 year old software engineer living in Bangalore."
})

# Step 7: Access as object
print(result.name)        # John Doe
print(result.age)         # 35
print(result.occupation)  # Software Engineer
print(result.city)        # Bangalore
print(type(result))       # <class '__main__.PersonInfo'>