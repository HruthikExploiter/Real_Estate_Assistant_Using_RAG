# importing the relevant libraries
from uuid import uuid4
import streamlit as st
from pathlib import Path
import shutil
from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Get API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# for locall purpose
#load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "word_embedding_chunks"

# Global Components
llm = None
vector_store = None
embeddings = None


def initialize_components():
    """
    Initializes LLM and loads existing FAISS vector store if available.
    """
    global llm, vector_store, embeddings

    if llm is None:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,  #it is used in the cloud to secure our key
            model="llama3-70b-8192",
            temperature=0.9,
            max_tokens=500
        )

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

    if vector_store is None and VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        vector_store = FAISS.load_local(str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True)


def reset_vector_store():
    """
    Resets the FAISS vector store by deleting persistent index files.
    """
    global vector_store
    vector_store = None  # Clear in-memory reference

    if VECTORSTORE_DIR.exists():
        try:
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print("Warning: Failed to reset vectorstore directory:", e)


def process_urls(urls):
    """
    Scrapes data from URLs and stores processed documents into FAISS.
    """
    global vector_store, embeddings

    yield "🔧 Initializing components..."
    initialize_components()

    yield "📥 Loading data from URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "✂️ Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)

    yield f"💾 Adding {len(docs)} documents to FAISS database..."
    if vector_store:
        vector_store.add_documents(docs)
    else:
        vector_store = FAISS.from_documents(docs, embedding=embeddings)

    vector_store.save_local(str(VECTORSTORE_DIR))

    yield "✅ Vector DB update complete for URLs!"


def process_txt_files(file_paths):
    """
    Process .txt files and update the FAISS vector store.
    """
    global vector_store, embeddings

    yield "🔧 Initializing components..."
    initialize_components()

    documents = []
    for path in file_paths:
        if Path(path).suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
            documents.extend(loader.load())
        else:
            yield f"⚠️ Skipping unsupported file: {path}"

    yield "✂️ Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    yield f"💾 Adding {len(chunks)} chunks to FAISS database..."
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    vector_store.save_local(str(VECTORSTORE_DIR))

    yield "✅ Vector DB update complete for .txt files!"


def process_csv_files(file_paths):
    """
    Process .csv files and update the FAISS vector store.
    """
    global vector_store, embeddings

    yield "🔧 Initializing components..."
    initialize_components()

    documents = []
    for path in file_paths:
        if Path(path).suffix == ".csv":
            loader = CSVLoader(file_path=str(path), encoding="utf-8", csv_args={"delimiter": ","})
            documents.extend(loader.load())
        else:
            yield f"⚠️ Skipping unsupported file: {path}"

    yield "✂️ Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    yield f"💾 Adding {len(chunks)} chunks to FAISS database..."
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    vector_store.save_local(str(VECTORSTORE_DIR))

    yield "✅ Vector DB update complete for .csv files!"


def generate_answer(query):
    """
    Answers a query using the retrieval-augmented generation chain.
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized.")

    retriever = vector_store.as_retriever()

    document_chain = create_stuff_documents_chain(llm)

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    result = retrieval_chain.invoke({"input": query})

    answer = result.get("answer", "")
    sources = ""

    if "context" in result:
        sources = ", ".join(
            list(
                set(
                    doc.metadata.get("source", "Unknown")
                    for doc in result["context"]
                )
            )
        )

    return answer, sources



def get_answer(query: str) -> str:
    """
    Initializes components (if needed) and returns the answer for a given query.
    """
    initialize_components()
    try:
        answer, sources = generate_answer(query)
        if sources:
            return f"{answer}\n\n📚 Sources: {sources}"
        return answer
    except Exception as e:
        return f"❌ Error generating answer: {e}"


if __name__ == "__main__":
    input_type = "url"  # Change to "csv" or "txt" as needed

    reset_vector_store()  # Only reset once at start

    if input_type == "url":
        urls = [
            "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
            "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
        ]
        for update in process_urls(urls):
            print(update)

    elif input_type == "csv":
        csv_files = [
            "data/mortgage_data.csv"
        ]
        for update in process_csv_files(csv_files):
            print(update)

    elif input_type == "txt":
        txt_files = [
            "data/mortgage_article.txt"
        ]
        for update in process_txt_files(txt_files):
            print(update)

    answer, sources = generate_answer("Summarize the information about 30-year mortgage rates.")
    print(f"\n🧠 Answer: {answer}")
    print(f"📚 Sources: {sources}")
