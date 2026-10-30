import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load Environment Variables
load_dotenv()

# Make sure you only need this in your .env file now:
# GOOGLE_API_KEY=AIza...

# 2. Initialize the Gemini LLM (Upgraded to 3.8-flash)
print("⏳ Initializing Gemini 3.8-flash Model...")
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# 3. Load and Split the Data
print("📄 Loading and splitting data...")
loader = TextLoader("data.txt", encoding="utf-8") # Ensure this file exists!
docs = loader.load()

# Split text into chunks so the LLM can process it
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 4. Create Embeddings and Vector Store
print("🔢 Creating embeddings (this might take a moment)...")
# We use Gemini for embeddings as it's highly reliable and free
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Store the chunks in FAISS (local vector database)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

# 5. Define the RAG Prompt
prompt = PromptTemplate.from_template(
    """You are an expert writer. Write a detailed article about the topic requested.
Use ONLY the following context to generate your response. If the context does not 
contain information about the topic, say "I don't have enough information."

Context:
{context}

Topic: {question}

Article:
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Build the RAG Chain using LCEL
print("🔗 Building the RAG chain...")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 7. Run the Application
if __name__ == "__main__":
    topic = "Write a summary about the main points in the text." # Change this to your desired topic
    
    print(f"\n🤖 Generating response for topic: '{topic}'...\n")
    response = rag_chain.invoke(topic)
    
    print("="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(response)