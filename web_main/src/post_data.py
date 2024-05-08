import os
import json
import numpy as np
from preprocess.qdrant import VectorDB

CONFIG_PATH = "/home/prompt_eng/langchain/langchain_proto/web_main/config/config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

FORMAT_PATH = "/home/prompt_eng/langchain/langchain_proto/web_main/data/format"
FILE_PATH = "format1.txt"

with open(os.path.join(FORMAT_PATH, FILE_PATH), "r", encoding="utf-8") as f:
    context = f.read()

print(context)

qdrant = VectorDB()
db_client = qdrant.create_connection()
chunks = qdrant.get_chunks(text=context)

payload_dict = {"filter": "format"}

meta_datas = list(np.repeat(payload_dict, len(chunks)))
ids = list(range(len(chunks)))
print(meta_datas)
print(ids)

COLLECTION_NAME = "format"
qdrant.create_collection(qd_client=db_client, collection_name=COLLECTION_NAME)
qdrant.add_vectorstore(
    client=db_client,
    text_chunks=chunks,
    collection_name=COLLECTION_NAME,
    ids=ids,
    metadatas=meta_datas,
    # payload_dict=payload_dict,
)

# db_client.set_payload(
#     collection_name=collection_name,
#     payload=payload_dict,
#     points=ids
# )
