import json

from pathlib import Path


document_path = Path(
    "documents/emc_internal_guide.txt"
)

text = document_path.read_text(
    encoding="utf-8"
)


CHUNK_SIZE = 180
OVERLAP = 40


def chunk_text(text, chunk_size, overlap):

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        chunks.append(
            {
                "chunk_id": chunk_id,
                "source": document_path.name,
                "start": start,
                "end": end,
                "text": chunk
            }
        )

        chunk_id += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


chunks = chunk_text(
    text,
    CHUNK_SIZE,
    OVERLAP
)


print(
    f"原始文件長度：{len(text)} 個字元"
)

print(
    f"Chunk 數量：{len(chunks)}"
)


for chunk in chunks:

    print("\n" + "=" * 60)

    print(
        f"Chunk {chunk['chunk_id']}"
        f" | {chunk['start']} → {chunk['end']}"
    )

    print(
        f"來源：{chunk['source']}"
    )

    print("=" * 60)

    print(chunk["text"])


output_path = Path(
    "output/chunks.json"
)

output_path.write_text(
    json.dumps(
        chunks,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print(
    f"\nChunks 已儲存到：{output_path}"
)