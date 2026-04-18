# 🔍 RAG — Retrieval-Augmented Generation

A local RAG (Retrieval-Augmented Generation) pipeline built with **LangChain**, **HuggingFace Embeddings**, **FAISS**, and **Google Gemini** — managed entirely with **uv**.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances LLM responses by first retrieving relevant context from a knowledge base before generating an answer. Instead of relying solely on what the model was trained on, RAG grounds responses in your own documents.

```
User Query
    │
    ▼
[HuggingFace Embeddings]  ──►  Embed query
    │
    ▼
[FAISS Vector Store]  ──►  Retrieve top-k relevant chunks
    │
    ▼
[LangChain Chain]  ──►  Combine context + query into prompt
    │
    ▼
[Google Gemini (LLM)]  ──►  Generate grounded answer
```

---

## 🗂️ Project Structure

```
RAG/
├── data/               # Your source documents (PDFs, text files, etc.)
├── rag.py              # Main RAG pipeline script
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked dependency versions (managed by uv)
├── .python-version     # Pinned Python version (3.14+)
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Package Manager | [`uv`](https://github.com/astral-sh/uv) |
| LLM Framework | [`LangChain`](https://github.com/langchain-ai/langchain) |
| Embeddings | `HuggingFaceEmbeddings` via `langchain-huggingface` + `sentence-transformers` |
| Vector Store | `FAISS` (CPU) |
| LLM | Google Gemini via `langchain-google-genai` |
| Environment Vars | `python-dotenv` |
| Python | `>=3.14` |

---

## 🚀 Getting Started

### 1. Prerequisites

Install [`uv`](https://docs.astral.sh/uv/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/nsfwsabir/RAG.git
cd RAG
```

### 3. Install dependencies

`uv` reads `pyproject.toml` and resolves everything from `uv.lock` automatically:

```bash
uv sync
```

> No need to manually create a virtualenv — `uv` handles it.

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Add your documents

Place your source files (PDFs, `.txt`, etc.) inside the `data/` directory.

### 6. Run the pipeline

```bash
uv run rag.py
```

---

## 📦 Dependencies

Defined in `pyproject.toml`:

```toml
dependencies = [
    "dotenv>=0.9.9",
    "faiss-cpu>=1.13.2",
    "langchain>=1.2.15",
    "langchain-community>=0.4.1",
    "langchain-google-genai>=4.2.2",
    "langchain-huggingface>=1.2.2",
    "sentence-transformers>=5.4.1",
]
```

---

## 🔧 How It Works

1. **Document Loading** — Files from `data/` are loaded using LangChain document loaders.
2. **Chunking** — Documents are split into overlapping chunks using a `RecursiveCharacterTextSplitter`.
3. **Embedding** — Each chunk is embedded using `HuggingFaceEmbeddings` (backed by `sentence-transformers`), running fully **locally** — no API call needed for embeddings.
4. **Indexing** — Embeddings are stored in a **FAISS** in-memory vector store for fast similarity search.
5. **Retrieval** — On a user query, the top-k most relevant chunks are retrieved via cosine/L2 similarity.
6. **Generation** — The retrieved context is passed along with the query to **Google Gemini** via `langchain-google-genai` to produce a final answer.

---

## 🌐 Why uv?

[`uv`](https://github.com/astral-sh/uv) is a blazing-fast Python package and project manager written in Rust. It replaces `pip`, `pip-tools`, `virtualenv`, and `poetry` in a single tool:

- `uv sync` — installs all deps from `uv.lock` (reproducible)
- `uv run` — runs scripts inside the managed environment
- `uv add <package>` — adds a new dependency

---

## 📝 Notes

- Embeddings run **locally** via HuggingFace — no embedding API key required.
- FAISS index is **in-memory** by default; persist it with `faiss.write_index()` if needed across runs.
- Swap out Gemini for any other LangChain-compatible LLM (Ollama, OpenAI, Anthropic, etc.) by changing the LLM init in `rag.py`.

---

## 👤 Author

**Mohammad Sabir** — [@nsfwsabir](https://github.com/nsfwsabir)

---

## 📄 License

This project is open source. Feel free to use, fork, and build on it.
