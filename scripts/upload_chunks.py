import json

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct
)


client = QdrantClient(
    path="qdrant_data"
)

COLLECTION_NAME = "enterprise_knowledge"

INPUT_PATH = Path(
    "input/embedded_chunks.json"
)


chunks = json.loads(
    INPUT_PATH.read_text(
        encoding="utf-8"
    )
)


points = []


for chunk in chunks:

    point = PointStruct(
        id=chunk["chunk_id"],

        vector=chunk["embedding"],

        payload={
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"]
        }
    )

    points.append(point)


client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)


print(
    f"成功寫入 {len(points)} 個 Points！"
)