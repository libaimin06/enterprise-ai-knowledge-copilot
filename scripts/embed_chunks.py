import json

from pathlib import Path
from google import genai
from google.genai import types


client = genai.Client()

INPUT_PATH = Path(
    "input/chunks.json"
)

OUTPUT_PATH = Path(
    "output/embedded_chunks.json"
)


chunks = json.loads(
    INPUT_PATH.read_text(
        encoding="utf-8"
    )
)


def embed_document(text, title):

    prepared_text = (
        f"title: {title} | text: {text}"
    )

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=prepared_text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    return result.embeddings[0].values


for chunk in chunks:

    print(
        f"正在處理 Chunk {chunk['chunk_id']}..."
    )

    vector = embed_document(
        chunk["text"],
        chunk["source"]
    )

    chunk["embedding"] = vector

    print(
        f"完成：{len(vector)} 維"
    )


OUTPUT_PATH.write_text(
    json.dumps(
        chunks,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("全部完成！")
print(
    f"檔案位置：{OUTPUT_PATH}"
)