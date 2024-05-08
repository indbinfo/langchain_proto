"""
* Version: 1.0
* 파일명: post_data.py
* 설명: 주어진 텍스트 데이터를 Qdrant에 벡터로 저장하는 모듈
* 수정일자: 2024/05/07
* 수정자: 손예선
* 수정 내용
    1. 환경에 따라 경로 설정
"""
import os
import json
import numpy as np
from preprocess.qdrant import VectorDB

# 운영 체제에 따라 경로 설정
if os.name == 'posix':
    PATH = os.path.join(
        os.environ.get('HOME'),
        'langchain_proto',
        'web_main',
    )
elif os.name == 'nt':
    PATH = os.path.join(
       'c:', 
       os.environ.get('HOMEPATH'),
       "langchain_proto", 
       "web_main",
    )
else:
    PATH = None

with open (os.path.join(PATH, 'config', 'config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)
with open(os.path.join('data', 'format', 'format1.txt'),'r', encoding='utf-8') as f:
    context = f.read()
print(context)

qdrant = VectorDB()
db_client = qdrant.create_connection()
chunks = qdrant.get_chunks(text = context)

payload_dict = {"filter":"format"}

meta_datas = list(np.repeat(payload_dict,len(chunks)))
ids=list(range(len(chunks)))
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
)
