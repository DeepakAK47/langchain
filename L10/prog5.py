# website loader
import os
from unittest import loader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
docs = loader.load()

print(f"Loaded {len(docs)} documents")
print(f"Metadata: {docs[0].metadata}")
print(f"Content preview: {docs[0].page_content[:3000]}")