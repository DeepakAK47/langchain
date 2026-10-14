# LangChain Learning Journey

A structured, hands-on repository documenting my learning path through LangChain — from basic LLM interactions to building RAG pipelines and advanced text processing.

---

## 📚 Table of Contents

- [L3 — LLM & Chat Model Basics](#l3--llm--chat-model-basics)
- [L4 — Chatbot & Streamlit UI](#l4--chatbot--streamlit-ui)
- [L5 — Structured Data with TypedDict & Pydantic](#l5--structured-data-with-typeddict--pydantic)
- [L6 — Output Parsers](#l6--output-parsers)
- [L7 — Chains](#l7--chains)
- [L8 & L9 — Runnables](#l8--l9--runnables)
- [L10 — Document Loaders](#l10--document-loaders)
- [L11 — Text Splitters](#l11--text-splitters)
- [Tech Stack](#-tech-stack)
- [Setup](#-setup)

---

## L3 — LLM & Chat Model Basics

- Interact with LLMs (completion-style models)
- Interact with ChatModels (message-based models)
- Interact with Hugging Face models via API

---

## L4 — Chatbot & Streamlit UI

- Built a conversational chatbot using Gemini API
- Built a Streamlit UI for summarizing input text

---

## L5 — Structured Data with TypedDict & Pydantic

- Used `TypedDict` to convert unstructured data into dictionary format
- Used `Pydantic` to validate and structure data
- Compared JSON Schema vs Pydantic Schema for structured output

---

## L6 — Output Parsers

- `JsonOutputParser` — parse JSON responses using chains
- `PydanticOutputParser` — validate and parse into Pydantic models
- `StrOutputParser` — extract plain string from `AIMessage`
- Structured parser using chains

---

## L7 — Chains

- Simple chain
- Sequential chain
- Conditional chain
- Parallel chain

---

## L8 & L9 — Runnables

- `RunnablePassthrough` — forward input unchanged
- `RunnableSequence` — chain Runnables sequentially
- `RunnableParallel` — run multiple Runnables concurrently
- `RunnableLambda` — wrap custom Python functions as Runnables
- `RunnableBranch` — conditional routing between chains

---

## L10 — Document Loaders

- `TextLoader` — load `.txt` files
- `PyPDFLoader` — load `.pdf` files with page metadata
- `WordLoader` — load `.docx` files
- `HTMLLoader` — load HTML pages

---

## L11 — Text Splitters

- **Length-based** — split by character or token count
- **Text-based** — split by paragraph/sentence boundaries
- **Document-based** — split by headings (Markdown, HTML, code)
- **Semantic-based** — split by meaning using embeddings

---

## 🛠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.11 |
| **Framework** | LangChain (core, community, experimental) |
| **LLM Provider** | Google Gemini API |
| **Embeddings** | `gemini-embedding-001` |
| **Vector Store** | FAISS |
| **UI** | Streamlit |
| **Validation** | Pydantic, TypedDict |
| **Environment** | python-dotenv |

---
