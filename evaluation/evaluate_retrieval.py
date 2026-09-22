import json
import csv

from pathlib import Path

from google import genai
from google.genai import types

from qdrant_client import QdrantClient


# =========================
# Client
# =========================

genai_client = genai.Client()

qdrant_client = QdrantClient(
    path="qdrant_data"
)


COLLECTION_NAME = "enterprise_knowledge"

TOP_K = 3


# =========================
# Embedding
# =========================

def embed_query(question):

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
# Retrieval
# =========================

def retrieve(question):

    query_vector = embed_query(
        question
    )

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True
    ).points

    return results


# =========================
# Load Dataset
# =========================

dataset_path = Path(
    "eval_dataset.json"
)

dataset = json.loads(
    dataset_path.read_text(
        encoding="utf-8"
    )
)


# =========================
# Evaluation
# =========================

total_answerable = 0
retrieval_hits = 0

evaluation_rows = []


for item in dataset:

    question = item["question"]

    results = retrieve(
        question
    )

    retrieved_ids = [
        point.payload["chunk_id"]
        for point in results
    ]

    top_score = (
        results[0].score
        if results
        else None
    )

    hit = None

    if item["answerable"]:

        total_answerable += 1

        expected_ids = set(
            item["expected_chunk_ids"]
        )

        retrieved_set = set(
            retrieved_ids
        )

        hit = bool(
            expected_ids &
            retrieved_set
        )

        if hit:
            retrieval_hits += 1

    print("\n" + "=" * 70)

    print(
        f"題目 {item['id']}："
        f"{question}"
    )

    print(
        "可回答：",
        item["answerable"]
    )

    print(
        "Retrieval Chunk IDs：",
        retrieved_ids
    )

    print(
        "Top Score：",
        round(top_score, 4)
        if top_score is not None
        else None
    )

    print(
        "Hit：",
        hit
    )


    evaluation_rows.append(
        {
            "id": item["id"],
            "question": question,
            "answerable": item["answerable"],
            "expected_chunk_ids": (
                str(
                    item["expected_chunk_ids"]
                )
            ),
            "retrieved_chunk_ids": (
                str(retrieved_ids)
            ),
            "top_score": (
                round(top_score, 4)
                if top_score is not None
                else None
            ),
            "retrieval_hit": hit
        }
    )


# =========================
# Metrics
# =========================

hit_rate = (
    retrieval_hits /
    total_answerable
    if total_answerable > 0
    else 0
)


print("\n")
print("=" * 70)
print("Evaluation Result")
print("=" * 70)

print(
    "Answerable Questions：",
    total_answerable
)

print(
    "Retrieval Hits：",
    retrieval_hits
)

print(
    "Hit@3：",
    round(
        hit_rate * 100,
        2
    ),
    "%"
)


# =========================
# CSV
# =========================

output_path = Path(
    "retrieval_results.csv"
)

with output_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=
        evaluation_rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        evaluation_rows
    )


print(
    "\n結果已輸出：",
    output_path
)