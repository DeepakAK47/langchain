# use of gemini chat model
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
result = model.invoke("Who is Virat kohli?")
print(result.content)
