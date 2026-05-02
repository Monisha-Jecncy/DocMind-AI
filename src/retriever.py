from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def get_retriever(file_name=None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    if file_name:
        return db.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "filter": {"source": file_name}}
        )

    return db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
