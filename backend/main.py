import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types, errors
from qdrant_client import QdrantClient


# =========================
# 基本設定
# =========================

app = FastAPI(
    title="Enterprise AI Knowledge Copilot",
    description="Gemini + Qdrant RAG API",
    version="0.3.0"
)

genai_client = genai.Client()

qdrant_client = QdrantClient(
    path="qdrant_data"
)

COLLECTION_NAME = "enterprise_knowledge"

TOP_K = 3


# =========================
# Request Schema
# =========================

class AskRequest(BaseModel):
    question: str


# =========================
# Query Embedding
# =========================

def embed_query(question: str):

    prepared_query = (
        f"task: question answering | query: {question}"
    )

    result = genai_client.models.embed_content(
        model="gemini-embedding-2",
        contents=prepared_query,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    return result.embeddings[0].values


# =========================
# Qdrant Retrieval
# =========================

def retrieve_chunks(question: str):

    query_vector = embed_query(question)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True
    ).points

    return results


# =========================
# Context Builder
# =========================

def build_context(results):

    context_parts = []

    for rank, point in enumerate(
        results,
        start=1
    ):

        payload = point.payload

        context_part = f"""
【參考資料 {rank}】

來源：
{payload["source"]}

Chunk ID：
{payload["chunk_id"]}

內容：
{payload["text"]}
"""

        context_parts.append(
            context_part
        )

    return "\n".join(
        context_parts
    )


# =========================
# Gemini Answer
# =========================

def generate_answer(
    question: str,
    context: str
):

    prompt = f"""
你是一個企業內部知識助理。

請根據提供的參考資料回答使用者問題。

規則：

1. 只能依照參考資料回答。
2. 不要用自己的外部知識補充。
3. 不要自行推測。
4. 如果參考資料不足以回答，
   請回答：
   「目前文件中沒有足夠資訊。」
5. 回答請簡潔清楚。
6. 回答內容若有依據，
   請標示來源文件名稱。

=========================
參考資料
=========================

{context}

=========================
使用者問題
=========================

{question}
"""

    max_retries = 3

    for attempt in range(
        max_retries
    ):

        try:

            response = (
                genai_client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
            )

            return response.text

        except errors.ServerError as e:

            print(
                f"Gemini Server Error "
                f"第 {attempt + 1} 次"
            )

            print(e)

            if attempt < max_retries - 1:

                time.sleep(2)

            else:

                raise HTTPException(
                    status_code=503,
                    detail="Gemini 目前服務繁忙，請稍後再試。"
                )


# =========================
# API
# =========================

@app.get("/")
def home():

    return {
        "status": "ok",
        "version": "0.3.0",
        "message": "RAG API is running"
    }


@app.post("/ask-rag")
def ask_rag(
    request: AskRequest
):

    try:

        # Step 1
        # Retrieval
        results = retrieve_chunks(
            request.question
        )

        # Step 2
        # 建立 Context
        context = build_context(
            results
        )

        # Step 3
        # Gemini 回答
        answer = generate_answer(
            request.question,
            context
        )

        # Step 4
        # 整理來源
        sources = []

        for point in results:

            payload = point.payload

            sources.append(
                {
                    "source": payload["source"],
                    "chunk_id": payload["chunk_id"],
                    "score": round(
                        point.score,
                        4
                    )
                }
            )

        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            type(e).__name__,
            e
        )

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )