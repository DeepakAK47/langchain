# Length based(RecursiveCharacterTextSplitter)
from langchain_text_splitters import RecursiveCharacterTextSplitter
text = "''Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# RecursiveCharacterTextSplitter → 100 characters
# TokenTextSplitter → 100 tokens
# chunk_overlap=20
# What it means: Each chunk shares 20 units with the previous chunk

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")