from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


client = QdrantClient(
    path="qdrant_data"
)

COLLECTION_NAME = "enterprise_knowledge"


if not client.collection_exists(
    COLLECTION_NAME
):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

    print("Collection 建立成功！")

else:

    print("Collection 已存在！")


print(
    client.get_collection(
        COLLECTION_NAME
    )
)