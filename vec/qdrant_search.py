from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

import os
import numpy as np
import openai
import tiktoken
import json

with open('/home/vec/main/vec/config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

os.environ['QDRANT_HOST'] = config['QDRANT_HOST']
os.environ['QDRANT_API_KEY'] = config['QDRANT_API_KEY']
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# calculate token length

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# split texts into chunks

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# embedding model definition(be aware of dimension size!!!)

def get_embedding():
    embeddings = HuggingFaceEmbeddings(
                                    model_name="jhgan/ko-sroberta-multitask",
                                    model_kwargs={'device': 'cpu'},
                                    encode_kwargs={'normalize_embeddings': True}
                                    )  
    return embeddings

# transform chucnks into vector and add to vectorstore

def add_vectorstore(text_chunks, collection_name,ids, metadatas
                    ):
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=get_embedding()
    )
    vectorstore.add_texts(text_chunks
                        ,ids=ids
                        ,metadatas=metadatas
                        )
    return vectorstore

# create query filter

def get_filter(key, value):
    filter = models.Filter(
    must=[models.FieldCondition(key=key
                                , match=models.MatchValue(value=value))]
                                )
    return filter

# llm definition

def load_llm():
    llm = OpenAI(openai_api_key= config['OPENAI_API_KEY'])
    #llm = CTransformers(
        #model = "TheBloke/Llama-2-7B-Chat-GGML",
        #model_type="llama",
        #temperature = 0.2)
    return llm

# plug the vectorstore to retrieval chain

def qdrant_qa_response(task, collection_name):
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    vectordb = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=get_embedding()
        )
    llm = load_llm()
    qa = RetrievalQA.from_chain_type(llm=load_llm(),
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True
                                       #,chain_type_kwargs={'prompt': prompt}
                                       )

    response = qa.invoke({'query': task})

    return response

# Similarity search with score by filter

def qdrant_similarity_search(task, collection_name, filter):
    client = qdrant_client.QdrantClient(
                os.getenv("QDRANT_HOST"),
                api_key=os.getenv("QDRANT_API_KEY")
                )
    vectordb = Qdrant(client=client,
                collection_name=collection_name,
                embeddings=get_embedding()
                )
    search_result = vectordb.similarity_search_with_score(task 
                                        #, k=k
                                        #, score_threshold =0.3         
                                        , filter=filter)
    return search_result

