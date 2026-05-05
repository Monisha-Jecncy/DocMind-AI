from langchain_openai import ChatOpenAI
from src.retriever import get_retriever


def ask_bot(query, file_name=None):
    try:
        print("Query:", query)
        retriever = get_retriever(file_name)
        docs = retriever.invoke(query)
        print("Docs found:", len(docs))

        if not docs:
            return {"answer": "Not available in the document", "sources": []}

        context = "\n\n".join([doc.page_content for doc in docs])

        sources = []
        for doc in docs:
            sources.append(
                {"file": doc.metadata.get("source"), "page": doc.metadata.get("page")}
            )

        
        llm = ChatOpenAI(model="gpt-5-nano",temperature=1)

    
        response = llm.invoke(f"""
Answer ONLY from context below.

Context:
{context}

Question:
{query}
""")
        print("Response:", response)

        
        answer = response.content if hasattr(response, "content") else str(response)

        
        return {"answer": answer, "sources": sources}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"answer": str(e), "sources": []}
