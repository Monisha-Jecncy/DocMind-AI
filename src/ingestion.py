import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def ingest_documents():
    os.makedirs("data", exist_ok=True)

    docs = []

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            path = os.path.join("data", file)

            loader = PyPDFLoader(path)
            pages = loader.load()

            for i, doc in enumerate(pages):
                doc.metadata["page"] = i + 1
                doc.metadata["source"] = file

            docs.extend(pages)

    if not docs:
        print("⚠️ No PDFs found")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    chunks = splitter.split_documents(docs)

    
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory="chroma_db"
    )

    db.persist()
    print("✅ ChromaDB created")
if __name__ == "__main__":
    ingest_documents()