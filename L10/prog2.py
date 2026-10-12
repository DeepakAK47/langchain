import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Chat with Your Text File")

# ---------- Sidebar: File upload ----------
with st.sidebar:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

# ---------- Build chain once per uploaded file ----------
@st.cache_resource(show_spinner="Building knowledge base...")
def build_chain(file_content: str):
    # Save uploaded content to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write(file_content)
        temp_path = f.name

    # Step 1: Load
    loader = TextLoader(temp_path, encoding="utf-8")
    docs = loader.load()

    # Step 2: Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Step 3: Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Step 4: Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 5: LLM
    model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))

    # Step 6: Prompt
    prompt = PromptTemplate.from_template(
        """Answer the question based only on the following context:

{context}

Question: {question}
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Step 7: Chain
    chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# ---------- Main area: Chat ----------
if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    chain = build_chain(file_content)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if query := st.chat_input("Ask a question about the file..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(query)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload a .txt file from the sidebar to start chatting.")