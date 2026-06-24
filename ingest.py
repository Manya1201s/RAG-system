import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

UPLOAD_FOLDER = "uploads"
CHROMA_DB_DIR = "chroma_db"

def ingest_documents():
    documents = []

    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(UPLOAD_FOLDER, file)

            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    if not documents:
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    return True


if __name__ == "__main__":
    ingest_documents()