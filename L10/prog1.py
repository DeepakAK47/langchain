# use of textloader(Document Loader)
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sampleData.txt")
docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(f"Content:\n{docs[0].page_content}")
print(f"Metadata: {docs[0].metadata}")