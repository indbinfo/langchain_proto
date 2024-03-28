from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import qdrant_client
import numpy as np
import json

# config
with open('/home/llm/main/llm/config/qdrant.json', 'r', encoding='utf8') as f:
    config = json.load(f)

QDRANT_HOST = config['QDRANT_HOST']
QDRANT_API_KEY = config['QDRANT_API_KEY']

client = QdrantClient(
    QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

context_path = '/home/llm/main/llm/context/'

class VectorDB:
    def __init__(self, client, collection_name, size=768):
        self.client = client
        self.collection_name = collection_name
        self.size=size
    
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
            collection_name = self.collection_name,
            vectors_config = collection_config
        )

    def add_vectorstore(self, text_chunks, ids, metadatas):
        vectorstore = Qdrant(
            client = self.client,
            collection_name = self.collection_name,
            embeddings = self.get_embedding()
        )
        vectorstore.add_texts(text_chunks,
                             ids=ids,
                             metadatas=metadatas
                             )
        return vectorstore

if __name__ == "__main__":
    collection_name = 'question'
    vectordb = VectorDB(collection_name=collection_name, size=768, client=client)

    vectordb.create_collection()

    with open(context_path + "valid.txt", 'rt', encoding='UTF8') as f:
        valid = f.read()

    with open(context_path + 'invalid.txt', 'rt', encoding='UTF8') as f:
        invalid = f.read()

    valid_list = valid.split('\n')
    invalid_list = invalid.split('\n')

    meta_1=list(np.repeat({"filter":"무효질문"},len(invalid_list)))
    meta_2=list(np.repeat({"filter":"유효질문"},len(valid_list)))

    question_list = invalid_list + valid_list
    ids = list(range(len(meta_1) + len(meta_2)))

    # vectorstore 저장
    vec = vectordb.add_vectorstore(
        text_chunks=question_list, 
        ids=ids, 
        metadatas=sum([meta_1,meta_2] , [])
        )

    # payload 설정
    client.set_payload(
        collection_name=collection_name,
        payload={'filter': '무효질문'},
        points=list(range(0, len(invalid_list)))
    )

    client.set_payload(
        collection_name=collection_name,
        payload={'filter' : '유효질문'},
        points=list(range(len(invalid_list), len(invalid_list + valid_list)))
    )