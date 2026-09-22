import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types, errors

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
QDRANT_DIR = PROJECT_ROOT / "qdrant_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# =========================================================
# Configuration
# =========================================================

COLLECTION_NAME = "enterprise_knowledge"

EMBEDDING_MODEL = "gemini-embedding-2"

VECTOR_SIZE = 768

CHUNK_SIZE = 180

CHUNK_OVERLAP = 40

MAX_RETRIES = 3


# =========================================================
# API Key Check
# =========================================================

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "找不到 GEMINI_API_KEY。\n"
        "請先設定 Gemini API Key 環境變數。"
    )


genai_client = genai.Client()


# =========================================================
# Chunking
# =========================================================

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    將文字切成具有 overlap 的 chunks。
    """

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP 必須小於 CHUNK_SIZE"
        )

    chunks = []

    start = 0
    local_chunk_id = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk_text_value = (
            text[start:end]
            .strip()
        )

        if chunk_text_value:

            chunks.append(
                {
                    "local_chunk_id": local_chunk_id,
                    "source": source,
                    "start": start,
                    "end": end,
                    "text": chunk_text_value,
                }
            )

            local_chunk_id += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =========================================================
# Load Documents
# =========================================================

def load_documents():
    """
    讀取 documents/ 內所有 txt 文件。
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"找不到 documents 資料夾："
            f"{DOCUMENTS_DIR}"
        )

    document_paths = sorted(
        DOCUMENTS_DIR.glob("*.txt")
    )

    if not document_paths:
        raise RuntimeError(
            "documents/ 裡沒有任何 .txt 文件。"
        )

    all_chunks = []

    print()
    print("======================================")
    print("讀取企業文件")
    print("======================================")

    for document_path in document_paths:

        print(
            f"\n讀取：{document_path.name}"
        )

        text = document_path.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(
            text=text,
            source=document_path.name,
        )

        print(
            f"產生 {len(chunks)} 個 chunks"
        )

        all_chunks.extend(
            chunks
        )

    return (
        document_paths,
        all_chunks,
    )


# =========================================================
# Embedding
# =========================================================

def embed_document(
    text: str,
    source: str,
):
    """
    將 Document Chunk 轉成 embedding vector。
    """

    prepared_text = (
        f"title: {source} | "
        f"text: {text}"
    )

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            result = (
                genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=prepared_text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=VECTOR_SIZE
                    ),
                )
            )

            vector = (
                result.embeddings[0].values
            )

            if len(vector) != VECTOR_SIZE:
                raise RuntimeError(
                    f"Embedding 維度錯誤："
                    f"{len(vector)}"
                )

            return vector

        except errors.ServerError as e:

            print(
                f"Gemini 暫時不可用 "
                f"({attempt + 1}/{MAX_RETRIES})"
            )

            print(e)

            if attempt < MAX_RETRIES - 1:
                time.sleep(2)

            else:
                raise


# =========================================================
# Qdrant
# =========================================================

def create_qdrant_collection():
    """
    重建 Qdrant collection。
    """

    client = QdrantClient(
        path=str(QDRANT_DIR)
    )

    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            f"\n刪除舊 Collection："
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    print(
        f"\n建立 Collection："
        f"{COLLECTION_NAME}"
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    return client


# =========================================================
# Build Index
# =========================================================

def build_index():

    print()
    print("======================================")
    print("Enterprise AI Knowledge Copilot")
    print("Vector Index Builder")
    print("======================================")

    document_paths, chunks = (
        load_documents()
    )

    print()
    print("======================================")
    print("產生 Embeddings")
    print("======================================")

    points = []

    embedded_chunks = []

    for global_id, chunk in enumerate(
        chunks
    ):

        print(
            f"[{global_id + 1}/{len(chunks)}] "
            f"{chunk['source']} "
            f"Chunk {chunk['local_chunk_id']}"
        )

        vector = embed_document(
            text=chunk["text"],
            source=chunk["source"],
        )

        point = PointStruct(
            id=global_id,
            vector=vector,
            payload={
                "chunk_id": global_id,
                "local_chunk_id": (
                    chunk["local_chunk_id"]
                ),
                "source": chunk["source"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
            },
        )

        points.append(
            point
        )

        embedded_chunks.append(
            {
                "chunk_id": global_id,
                "local_chunk_id": (
                    chunk["local_chunk_id"]
                ),
                "source": chunk["source"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
            }
        )

    qdrant_client = (
        create_qdrant_collection()
    )

    print()
    print("======================================")
    print("寫入 Qdrant")
    print("======================================")

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    count_result = (
        qdrant_client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )
    )

    print(
        f"成功寫入："
        f"{count_result.count} Points"
    )


    # =====================================================
    # Build Report
    # =====================================================

    report = {
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "vector_size": VECTOR_SIZE,
        "distance": "COSINE",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": len(
            document_paths
        ),
        "chunk_count": len(
            chunks
        ),
        "documents": [
            path.name
            for path in document_paths
        ],
    }

    report_path = (
        OUTPUT_DIR /
        "build_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    chunks_path = (
        OUTPUT_DIR /
        "chunks_manifest.json"
    )

    chunks_path.write_text(
        json.dumps(
            embedded_chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()
    print("======================================")
    print("Index Build 完成 ✅")
    print("======================================")

    print(
        f"Documents："
        f"{len(document_paths)}"
    )

    print(
        f"Chunks："
        f"{len(chunks)}"
    )

    print(
        f"Qdrant Points："
        f"{count_result.count}"
    )

    print(
        f"Collection："
        f"{COLLECTION_NAME}"
    )

    print(
        f"Build Report："
        f"{report_path}"
    )

    print()
    print(
        "接下來可以啟動 FastAPI Backend。"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    build_index()
