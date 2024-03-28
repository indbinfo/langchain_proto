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
        client = QdrantClient(self.host, api_key=self.api_key)
        return client
    def create_collection(self,qd_client, collection_name):
        collection_config = qdrant_client.http.models.VectorParams(
        size=768, # 1536 for OpenAI #768 for Gemini/HuggingFace/instructor-xl
        distance=qdrant_client.http.models.Distance.COSINE)
        qd_client.recreate_collection(
                                   collection_name=collection_name,
                                   vectors_config=collection_config
                                  )


    def create_Qdrant_obj(self,client, collection_name):
        qdrant_obj = Qdrant(client=client,
                    collection_name=collection_name,
                    embeddings=self.embeddings
                    )
        return qdrant_obj
    
    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)
    def get_chunks_test(self, text,chunk_size,chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.tiktoken_len,
            add_start_index=True
        )
        chunks = text_splitter.split_text(text)
        return chunks
    def get_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=self.tiktoken_len,
            add_start_index=True
        )
        chunks = text_splitter.split_text(text)
        return chunks
    def add_vectorstore(self, client, text_chunks, collection_name,ids, metadatas):
    
        # metadatas = list(np.repeat(payload_dict,len(text_chunks)))
        vectordb = self.create_Qdrant_obj(client, collection_name)
        vectordb.add_texts(
                            text_chunks,
                            ids=ids,
                            metadatas=metadatas
                          )
        # client.set_payload(
        #                     collection_name=collection_name,
        #                     payload=payload_dict,
        #                     points=ids
        #                   )
    #def add_vectorstore(self, client, text_chunks, collection_name,ids, metadatas,payload_dict):
    # Similarity search with score by filter
    def qdrant_similarity_search(self, client, task, collection_name, filter=None, k=1):
        vectordb = self.create_Qdrant_obj(client, collection_name)
        search_result = vectordb.similarity_search_with_score(query=task 
                                            , k=k       
                                            , filter=filter)
        return search_result


    
    def create_retriever(self,qd_obj, collection_name, filter_dict):
        retriever = qd_obj.as_retriever(
            search_kwargs=dict(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value)
                        )
                    ]
                )
            )
        )

def get_filter(key, value):
    filter = models.Filter(
    must=[models.FieldCondition(key=key
                                , match=models.MatchValue(value=value))]
                                )
    return filter

def format_docs(docs):
	# return docs[0].page_content
    return " ".join([doc.page_content for doc in docs])

def remove_overlaps_in_sequence(texts):
    """
    Removes overlapping parts from a list of texts where each text overlaps with the next one.
    Returns a single merged text with all overlaps removed.
    """
    if not texts:
        return ""
    # Start with the first text
    merged_text = texts[0]

    # Iterate through the texts, starting from the second one
    for next_text in texts[1:]:
        # Finding the overlap
        min_length = min(len(merged_text), len(next_text))
        for i in range(min_length, 0, -1):
            if merged_text.endswith(next_text[:i]):
                # Overlap found, concatenate the unique part of the merged_text with the unique part of next_text
                merged_text += next_text[i:]
                break
        else:
            # No overlap found, concatenate the whole next_text
            merged_text += next_text

    return merged_text