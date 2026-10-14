# sementic based splitting
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

# 1. Initialize the Gemini Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 2. Create the Semantic Chunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile", # Splits at the 95th percentile of differences
    # breakpoint_threshold_amount=95
)

# 3. Your text with distinct topics
text = """
Farmers were working hard in the fields, preparing the soil and planting seeds.
The sun was bright, and the air smelled of earth.

The Indian Premier League (IPL) is the biggest cricket league in the world.
People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear.
To fight terrorism, we need strong laws and alert security forces.
"""

# 4. Split the text
docs = text_splitter.create_documents([text])

# 5. Display the chunks
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print()