from qdrant_client import QdrantClient
import qdrant_client
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

import os
import numpy as np
import openai
import tiktoken
import json

with open('/home/prompt_eng/langchain/langchain_proto/web_main/config/qdrant.json', 'r', encoding='utf8') as f:
    config = json.load(f)

QDRANT_HOST = config['QDRANT_HOST']
QDRANT_API_KEY = config['QDRANT_API_KEY']



class VectorDB:
    def __init__(self):
        self.host = QDRANT_HOST
        self.api_key = QDRANT_API_KEY
        self.embeddings = HuggingFaceEmbeddings(
                                    model_name="jhgan/ko-sroberta-multitask",
                                    model_kwargs={'device': 'cpu'},
                                    encode_kwargs={'normalize_embeddings': True}
                                    )
    def create_client(self):
        client = QdrantClient(host=self.host, api_key=self.api_key)
        return client
    
    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    def get_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200,
            length_function=self.tiktoken_len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def add_vectorstore(self, text_chunks, collection_name,ids, metadatas):
        vectorstore = QdrantClient(
            client=client,
            collection_name=collection_name,
            embeddings=get_embedding()
        )
        vectorstore.add_texts(text_chunks
                            ,ids=ids
                            ,metadatas=metadatas
                            )
        return vectorstore

# Similarity search with score by filter
def qdrant_similarity_search(task, collection_name, filter):
    client = qdrant_client.QdrantClient(
                os.getenv("QDRANT_HOST"),
                api_key=os.getenv("QDRANT_API_KEY")
                )
    vectordb = QdrantClient(client=client,
                collection_name=collection_name,
                embeddings=get_embedding()
                )
    search_result = vectordb.similarity_search_with_score(task 
                                        #, k=k
                                        #, score_threshold =0.3         
                                        , filter=filter)
    return search_result
