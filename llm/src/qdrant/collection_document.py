from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import qdrant_client
import numpy as np
import tiktoken
import json
import sys

home_dir = '/home/llm/main/llm/'
config_path = home_dir + 'config/config.json'

# config
with open(config_path, 'r', encoding='utf8') as f:
    config = json.load(f)

context_path = home_dir + config['path']['context_path']

# qdrant config
with open(home_dir + 'config/qdrant.json', 'r', encoding='utf8') as f:
    qdrant = json.load(f)

QDRANT_HOST = qdrant['QDRANT_HOST']
QDRANT_API_KEY = qdrant['QDRANT_API_KEY']

client = QdrantClient(
    QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

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
    
    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)

    def get_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            separators = ['\n','|', ' '],
            keep_separator=True,
            chunk_size=400,
            chunk_overlap=40,
            length_function=self.tiktoken_len
        )
        chunks = text_splitter.split_text(text)
        return chunks

if __name__ == "__main__":
    collection_name = sys.argv[1]
    vectordb = VectorDB(collection_name=collection_name, size=768, client=client)

    vectordb.create_collection()

    with open(context_path + "1.txt", 'rt', encoding='UTF8') as f:
        prompt_1 = f.read()

    with open(context_path + "2.txt", 'rt', encoding='UTF8') as f:
        prompt_2 = f.read()

    with open(context_path + "3.txt", 'rt', encoding='UTF8') as f:
        prompt_3 = f.read()

    with open(context_path + "4.txt", 'rt', encoding='UTF8') as f:
        prompt_4 = f.read()

    with open(context_path + "5.txt", 'rt', encoding='UTF8') as f:
        prompt_5 = f.read()

    with open(context_path + "6.txt", 'rt', encoding='UTF8') as f:
        prompt_6 = f.read()
    
    meta_1 = [{"filter":"시간대"}]
    meta_2 = [{"filter":"추이"}]
    meta_3 = [{"filter":"마포구"}]
    meta_4 = [{"filter":"법인"}]
    meta_5 = [{"filter":"인천"}]
    meta_6 = [{"filter":"약국"}]

    vectordb.add_vectorstore(
    text_chunks=[prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6] ,
    ids = list(range(0,6)),
    metadatas=sum([meta_1, meta_2, meta_3, meta_4, meta_5, meta_6], [])
)
    client.set_payload(
    collection_name=collection_name,
    payload={'filter': '시간대'},
    points=[0]
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': '추이'},
    points=[1]
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': '마포구'},
    points=[2]
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': '법인'},
    points=[3]
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': '인천'},
    points=[4]
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': '약국'},
    points=[5]
    )