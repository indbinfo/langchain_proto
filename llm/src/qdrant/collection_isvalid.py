from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import qdrant_client
import numpy as np
import json

# config
with open('/home/llm/langchain_proto/web_main/config/qdrant.json', 'r', encoding='utf8') as f:
    config = json.load(f)

QDRANT_HOST = config['QDRANT_HOST']
QDRANT_API_KEY = config['QDRANT_API_KEY']

client = QdrantClient(
    QDRANT_HOST,
    api_key=QDRANT_API_KEY
)


class VectorDB:
    def __init__(self, client, collection_name, size=768):
        self.client = client
        self.collection_name = collection_name
        self.size = size

    # get embedding model
    def get_embedding(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings

    # create collection
    def create_collection(self):
        collection_config = qdrant_client.http.models.VectorParams(
            size=self.size,
            distance=qdrant_client.http.models.Distance.COSINE
        )

        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=collection_config
        )

    def add_vectorstore(self, text_chunks, ids, metadatas):
        vectorstore = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.get_embedding()
        )
        vectorstore.add_texts(text_chunks, ids=ids, metadatas=metadatas)
        return vectorstore


if __name__ == "__main__":
    collection_name = 'question'
    vectordb = VectorDB(collection_name=collection_name, size=768, client=client)

    vectordb.create_collection()

    # 유·무효질문
    with open('/home/llm/langchain_proto/vec/questions.json',
              'r',
              encoding='utf8') as f:
        questions = json.load(f)

    meta_1 = list(np.repeat({"filter": "무효질문"}, len(questions['무효질문'])))
    meta_2 = list(np.repeat({"filter": "유효질문"}, len(questions['유효질문'])))

    question_list = questions['무효질문'] + questions['유효질문']
    ids = list(range(len(meta_1) + len(meta_2)))

    # vectorstore 저장
    vec = vectordb.add_vectorstore(
        text_chunks=question_list,
        ids=ids,
        metadatas=sum([meta_1, meta_2], [])
        )

    # payload 설정
    client.set_payload(
        collection_name=collection_name,
        payload={'filter': '무효질문'},
        points=list(range(0, 39))
    )

    client.set_payload(
        collection_name=collection_name,
        payload={'filter': '유효질문'},
        points=list(range(39, 94))
    )
