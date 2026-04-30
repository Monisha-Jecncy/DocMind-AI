import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

os.environ["ANONYMIZED_TELEMETRY"] = "False"


def get_retriever(file_name=None):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )
    if file_name:
        return db.as_retriever(search_kwargs={"k": 8, "filter": {"source": file_name}})

    return db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})
