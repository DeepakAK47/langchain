from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#step1
client = genai.Client(api_key="")

# step2
documents = [
    "Java is an object-oriented programming language for cross-platform development.",
    "Java coffee is a bold Indonesian bean with low acidity.",
    "Java island is densely populated and home to Jakarta."
]
query = "What is Java programming language?"

# Create prefix functions
def prepare_document(content):
    """Add title prefix to help model understand this is a document."""
    return f"title: none | text: {content}"

def prepare_query(content):
    """Add task prefix to help model understand this is a question."""
    return f"task: question answering | query: {content}"

# STEP 3: Generate embeddings for documents
# ============================================
# Why: Each document gets converted to a vector for similarity comparison
doc_embeddings = []
for doc in documents:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[prepare_document(doc)],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",  # Tells API: "I'm embedding a document"
            output_dimensionality=768        # Reduces storage; 768 is a good balance
        )
    )
    doc_embeddings.append(response.embeddings[0].values)

# STEP 4: Generate embedding for query
# ============================================
# Why: Query vector is compared against all document vectors
query_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[prepare_query(query)],
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY"  # Tells API: "I'm embedding a question"
    )
)
query_embedding = query_response.embeddings[0].values


# ============================================
# STEP 5: Compute similarity scores
# ============================================
# Why: Cosine similarity measures how "close" each document is to the query
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# ============================================
# STEP 6: Rank and display results
# ============================================
# Why: Sort documents by similarity (highest first)
top_indices = similarities.argsort()[::-1]

print("\n=== Search Results ===")
print(f"Query: {query}\n")
for idx in top_indices:
    print(f"Score: {similarities[idx]:.4f} | Document: {documents[idx][:80]}...")