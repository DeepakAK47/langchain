# csv loader 
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import CSVLoader

load_dotenv()

loader = CSVLoader(file_path="insurance.csv")
docs = loader.load()

print(f"Loaded {len(docs)} rows")
print(docs[0].page_content)
print(docs[0].metadata)