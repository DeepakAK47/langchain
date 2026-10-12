# PYpdf Loader
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# step 1 Load pdf
loader = PyPDFLoader("Resume.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# Step 3: Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Step 4: Store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Step 6: Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Step 7: Prompt template
prompt = PromptTemplate.from_template(
    """Answer the question based only on the following context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
"""
)

# Step 8: Helper to format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 9: Build RAG chain
chain = (
    {"context": retriever | RunnableLambda(format_docs),
     "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Step 10: Ask questions
while True:
    query = input("\nAsk a question (or 'quit' to exit): ")
    if query.lower() == "quit":
        break
    answer = chain.invoke(query)
    print(f"\nAnswer: {answer}")
