import os
import shutil
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

os.environ["ANONYMIZED_TELEMETRY"] = "False"


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

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    chunks = splitter.split_documents(docs)
    if os.path.exists("chroma_db"):
        try:
            shutil.rmtree("chroma_db")
        except PermissionError:
            print("⚠️ DB in use, retrying...")
            time.sleep(2)
            shutil.rmtree("chroma_db")

    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
    client_settings=Settings(anonymized_telemetry=False)
)
    
    db.persist()


   


    print("✅ ChromaDB created")
if __name__ == "__main__":
    ingest_documents()
