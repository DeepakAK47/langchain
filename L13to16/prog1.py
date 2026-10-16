# Wekipedia Retrieval 
from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever()
docs = retriever.invoke("What is LangChain?")
print(docs[0].page_content)