# Real Estate Assistant Using RAG

Real Estate Assistant Using RAG is a smart assistant that leverages Retrieval-Augmented Generation (RAG) to provide intelligent, source-backed answers from uploaded documents or URLs. It combines the power of LangChain, HuggingFace embeddings, FAISS vector search, and LLMs (via Groq API) with a FastAPI backend.

![Screenshot](screenshot-1.png)

---

## App link
https://querybridge-smart-assistant.streamlit.app/

## 🔧 Features

- 📁 Accepts `.txt`, `.csv`, and web URLs for ingestion
- 🔍 Semantic search with FAISS vector store
- 🤖 LLM-powered answers using Groq's LLaMA 3 70B
- 🔄 Persistent vector database with metadata tagging
- 🌐 FastAPI for backend with CORS enabled
- 🧠 On-the-fly query answering with source references
- 🗑️ Auto-deletion of source-specific vectors

---

## 🗂️ Project Structure

```
Real_Estate_Assistant_Using_RAG-main/
│
├── FastAPI_Server/
│   └── server.py              # API endpoints for question-answering
│
├── model/
│   ├── main.py                # Core RAG logic (data ingestion, retrieval)
│   ├── rag.py                 # Embedding, FAISS store, processing logic
│   └── resources/vectorstore/ # Saved vector DB
│
├── requirements.txt           # Python dependencies
├── screenshot.png             # UI/UX visual reference
└── .env                       # API keys and environment variables
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
gh repo clone HruthikExploiter/Real_Estate_Assistant_Using_RAG
cd Real_Estate_Assistant_Using_RAG-main
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your `.env` file
Create a `.env` file in `model/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the App

### Backend (FastAPI)
```bash
cd FastAPI_Server
uvicorn server:app --reload
```

---

## 🧪 API Usage

### Health Check
`GET /`  
Response:
```json
{ "message": "FastAPI is running." }
```

### Ask a Question
`POST /ask`  
Request:
```json
{ "query": "What are 30-year mortgage rates?" }
```
Response:
```json
{
  "answer": "30-year mortgage rates are ...",
  "sources": "https://example.com"
}
```

---

## 📌 Notes

- The vector DB persists across runs unless reset.
- To delete source-specific content, call the `delete_documents_by_source()` function (e.g., inside a Streamlit app or separate API).
- All `.txt`, `.csv`, and URL data are chunked and embedded for semantic search.

---

## 📃 License

MIT License © 2025 — [Hruthik Gajjala]
