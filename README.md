# LangChain Learning Journey

A structured, hands-on repository documenting my learning path through LangChain — from basic LLM interactions to retrieval-augmented generation (RAG), vector stores, and tool calling.

The lessons are organized chronologically in folders named `L3` through `L17`. Each folder contains small Python examples focused on a specific LangChain concept.

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
- [L12 — Vector Stores with Chroma](#l12--vector-stores-with-chroma)
- [L13–L16 — Retrieval & Search](#l13l16--retrieval--search)
- [L17 — Tool Calling](#l17--tool-calling)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)

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

## L12 — Vector Stores with Chroma

- Create embeddings with `GoogleGenerativeAIEmbeddings`
- Store and retrieve document chunks with Chroma
- Persist a local vector database in `L12/chroma_db`
- Build a RAG chain with a retriever, prompt template, and Gemini

---

## L13–L16 — Retrieval & Search

- Retrieve information from Wikipedia with `WikipediaRetriever`
- Search the web with DuckDuckGo tools
- Explore retrieval workflows that provide external context to an LLM

---

## L17 — Tool Calling

- Define tools with the `@tool` decorator
- Bind custom tools to a Gemini chat model
- Handle model-generated tool calls and tool results
- Build examples for arithmetic, weather data, and currency conversion

---

## L18 — Modern Agent API

- Use the modern LangChain agent API
- Create agents with custom tools and system prompts
- Invoke agents with user messages
- Extract and display the final response


---

Python 3.11 is used throughout the project.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

The shared dependency list is available at [`L3/requirements.txt`](L3/requirements.txt).

```bash
pip install -r L3/requirements.txt
```

Some lessons use additional integrations, such as Chroma, community retrievers, Streamlit, or search tools. Install the relevant LangChain integration package if it is not already available in your environment.

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

The examples load this value with `python-dotenv`. Do not commit your `.env` file or API keys.

### 5. Run an example

```bash
python L17/prog2.py
python L12/prog1.py
```

The L12 example creates or updates its local Chroma data under `L12/chroma_db`.

---

## 📁 Repository Structure

```text
L3/       LLM and chat model integrations
L4/       Chatbot and Streamlit UI examples
L5/       TypedDict, JSON, and Pydantic structured output
L6/       Output parsers
L7/       Chains
L8and9/   Runnables
L10/      Document loaders
L11/      Text splitters
L12/      Chroma vector store and RAG
L13to16/  Retrieval and search
L17/      Tool calling
```

The repository also includes lesson-specific virtual environments and sample data. These are kept alongside the lessons for experimentation; a fresh environment is recommended for new setups.

---

## 🛠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.11 |
| **Framework** | LangChain |
| **LLM Provider** | Google Gemini API |
| **Embeddings** | `gemini-embedding-001` |
| **Vector Stores** | FAISS, Chroma |
| **Retrieval** | Wikipedia, DuckDuckGo |
| **UI** | Streamlit |
| **Validation** | Pydantic, TypedDict |
| **Environment** | python-dotenv |

---

## Notes

- Most examples are educational scripts and may require a configured API key or external service access.
- Run scripts from the repository root so relative paths, such as `L12/chroma_db`, resolve correctly.
- API model names and third-party integrations can change over time; check the relevant lesson before adapting an example for production use.
