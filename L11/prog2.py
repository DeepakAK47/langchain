# textStructure based Splitter
from langchain_text_splitters import MarkdownHeaderTextSplitter


# Your markdown text
markdown_text = """# LangChain Guide

## Introduction
LangChain is a framework for LLM applications.

## Document Loaders
Loaders read data from various sources.

### TextLoader
Reads plain text files.

### PyPDFLoader
Reads PDF files.

## Text Splitters
Splitters break text into chunks.
"""

# Define which headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Create splitter
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Split
chunks = splitter.split_text(markdown_text)

# Display results
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Content: {chunk.page_content}")
    print(f"Headers: {chunk.metadata}")
    