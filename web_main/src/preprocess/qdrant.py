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
    def create_connection(self):
        client = QdrantClient(host=self.host, api_key=self.api_key)
        return client
    def create_Qdrant_obj(self,client, collection_name):
        Qdrant(client=client,
                    collection_name=collection_name,
                    embeddings=self.embeddings
                    )
        return qdrant_obj
    
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

    def add_vectorstore(self, client, text_chunks, collection_name,ids, metadatas):
        vectordb = create_Qdrant_obj(self,client, collection_name)
        vectordb.add_texts(text_chunks
                            ,ids=ids
                            ,metadatas=metadatas
                            )
        return vectorstore
    # Similarity search with score by filter
    def qdrant_similarity_search(self, client, task, collection_name, filter):
        vectordb = create_Qdrant_obj(self,client, collection_name)
        search_result = vectordb.similarity_search_with_score(task 
                                            #, k=k
                                            #, score_threshold =0.3         
                                            , filter=filter)
        return search_result
