# Croma data store
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# Step 1: Sample documents (replace with your own)
documents = [
    "LangChain is a framework for building LLM applications.",
    "Chroma is an open-source vector database for AI applications.",
    "FAISS is a library for efficient similarity search.",
    "RAG combines retrieval with generation for accurate answers."
]

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(documents)

# Step 3: Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Step 4: Create Chroma vector store with persistence
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection",
    persist_directory="./chroma_db"  # Data saved here [citation:1][citation:14]
)

# Step 5: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 6: Initialize Gemini LLM
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Step 7: Prompt template
prompt = PromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 8: Build RAG chain
chain = (
    {"context": retriever | RunnableLambda(format_docs),
     "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Step 9: Ask questions
while True:
    query = input("\nAsk a question (or 'quit' to exit): ")
    if query.lower() == "quit":
        break
    answer = chain.invoke(query)
    print(f"\nAnswer: {answer}")