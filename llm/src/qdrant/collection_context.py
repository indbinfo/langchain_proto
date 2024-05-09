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

    chunks_1 = vectordb.get_chunks(prompt_1)
    chunks_2 = vectordb.get_chunks(prompt_2)
    chunks_3 = vectordb.get_chunks(prompt_3)
    chunks_4 = vectordb.get_chunks(prompt_4)
    chunks_5 = vectordb.get_chunks(prompt_5)
    chunks_6 = vectordb.get_chunks(prompt_6)
    chunck_list = sum([chunks_1,chunks_2, chunks_3, chunks_4, chunks_5, chunks_6],[] )
    
    meta_1 = list(np.repeat({"filter":"prompt_1"},len(chunks_1)))
    meta_2 = list(np.repeat({"filter":"prompt_2"},len(chunks_2)))
    meta_3 = list(np.repeat({"filter":"prompt_3"},len(chunks_3)))
    meta_4 = list(np.repeat({"filter":"prompt_4"},len(chunks_4)))
    meta_5 = list(np.repeat({"filter":"prompt_5"},len(chunks_5)))
    meta_6 = list(np.repeat({"filter":"prompt_6"},len(chunks_6)))

    vectordb.add_vectorstore(
    text_chunks=chunck_list ,
    ids = list(range(len(chunck_list))),
    metadatas=sum([meta_1, meta_2, meta_3, meta_4, meta_5, meta_6], [])
)
    client.set_payload(
    collection_name=collection_name,
    payload={'filter': 'prompt_1'},
    points=list(range(0,len(chunks_1)))
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': 'prompt_2'},
    points=list(range(len(chunks_1), len(chunks_1 + chunks_2)))
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': 'prompt_3'},
    points=list(range(len(chunks_1 + chunks_2), len(chunks_1 + chunks_2 + chunks_3)))
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': 'prompt_4'},
    points=list(range(len(chunks_1 + chunks_2 + chunks_3), len(chunks_1 + chunks_2 + chunks_3 + chunks_4)))
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': 'prompt_5'},
    points=list(range(len(chunks_1 + chunks_2 + chunks_3 + chunks_4), len(chunks_1 + chunks_2 + chunks_3 + chunks_4 + chunks_5)))
    )

    client.set_payload(
    collection_name=collection_name,
    payload={'filter': 'prompt_6'},
    points=list(range(len(chunks_1 + chunks_2 + chunks_3 + chunks_4 + chunks_5), len(chunks_1 + chunks_2 + chunks_3 + chunks_4 + chunks_5 + chunks_6)))
    )