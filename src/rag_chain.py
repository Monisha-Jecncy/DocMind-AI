from langchain_community.chat_models import ChatOllama
from src.retriever import get_retriever
from sentence_transformers import CrossEncoder

def ask_bot(query, file_name=None):
    try:
        retriever = get_retriever(file_name)

        docs = retriever.invoke(query)

        if not docs:
            return {"answer": "No data found", "sources": []}

        # 🔥 ADD THIS BLOCK (re-ranking)

        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        docs = docs[:3]  # keep top 3 best docs

        # 🔥 THEN build context
        context = "\n\n".join([doc.page_content for doc in docs])
        print("📄 Retrieved Context:\n", context)
        sources = [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0),
            }
            for doc in docs
        ]

        llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

        prompt = f"""
You are a document question-answering assistant.

Strict Rules:
- Answer ONLY using the given context.
- Do NOT assume, infer, or add external knowledge.
- If the answer is not clearly present, say:
  "Not available in the document."

Answer Style:
- Be clear, concise, and structured.
- Use bullet points for lists or classifications.
- Use exact values (numbers, percentages, dates) from the document.
- Do NOT include unnecessary explanations.

Priority:
- If multiple related details exist, choose the most relevant and direct answer.
- Prefer definitions, rules, or final values over examples or references.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        # FIX: handle response properly
        answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer.strip(), "sources": sources}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"answer": str(e), "sources": []}
