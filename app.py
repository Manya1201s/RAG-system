import streamlit as st
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from ingest import ingest_documents

UPLOAD_FOLDER = "uploads"
CHROMA_DB_DIR = "chroma_db"

st.set_page_config(
    page_title="Enterprise RAG System",
    layout="wide"
)

st.title("📚 Enterprise Knowledge Assistant")
st.caption("AI-Based Enterprise Knowledge Retrieval using RAG")

uploaded_files = st.file_uploader(
    "Drag & Drop PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        with open(
            os.path.join(UPLOAD_FOLDER, file.name),
            "wb"
        ) as f:
            f.write(file.getbuffer())

    st.success("Files Uploaded Successfully")

    if st.button("Process Documents"):
        with st.spinner("Creating Knowledge Base..."):
            ingest_documents()

        st.success("Knowledge Base Created")


question = st.text_input(
    "Ask a question about uploaded documents"
)

if st.button("Ask Question") and question:
    
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    docs = db.similarity_search(
        question,
        k=2
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer only from provided context.

    Context:
    {context}

    Question:
    {question}
    """

    llm = OllamaLLM(model="phi3")

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response)

    st.subheader("Sources")

    for doc in docs:
        st.write(
            doc.metadata.get(
                "source",
                "Unknown"
            )
        )