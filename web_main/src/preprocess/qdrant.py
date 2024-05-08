"""
* Version: 1.0
* 파일명: qdrant.py
* 설명: QdrantClient를 통해 벡터 컬렉션 생성, 벡터스토어에 추가, 유사도 검색을 수행하는 기능
* 수정일자: 2024/05/02
* 수정자: 손예선
* 수정 내용
    1. 사용하지 않은 모듈 Import 제외
    2. 환경에 따라 경로 설정
    3. 테스트 코드 삭제
    4. built-in 매개변수명 변경(filter -> filters)
"""
import os
import json

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
import qdrant_client
import qdrant_client.http
from qdrant_client.http import models

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


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

with open(os.path.join(PATH, 'config', 'qdrant.json'), encoding='utf-8') as f:
    config = json.load(f)

QDRANT_HOST = config['QDRANT_HOST']
QDRANT_API_KEY = config['QDRANT_API_KEY']


class VectorDB:
    """QdrantClient로 벡터 데이터를 저장하고 관리"""
    def __init__(self):
        self.host = QDRANT_HOST
        self.api_key = QDRANT_API_KEY
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_connection(self):
        """
        QdrantClient와 연결 생성
        
        Returns:
            QdrantClient: QdrantClient 연결 객체
        """
        client = QdrantClient(self.host, api_key=self.api_key)
        return client

    def create_collection(self, qd_client, collection_name):
        """ Collection 생성"""
        collection_config = qdrant_client.http.models.VectorParams(
        size=768, # 1536 for OpenAI # 768 for Gemini/HuggingFace/instructor-xl
        distance=qdrant_client.http.models.Distance.COSINE)
        qd_client.recreate_collection(
                                   collection_name=collection_name,
                                   vectors_config=collection_config
                                  )

    def create_qdrant_obj(self, client, collection_name):
        """
        Qdrant 객체 생성
        
        Args:
            client (QdrantClient): QdrantClient 연결 객체
            collection_name (str): 벡터 컬렉션 이름
        Returns:
            Qdrant: Qdrant 객체가 생성되어 반환
        """
        qdrant_obj = Qdrant(client=client,
                    collection_name=collection_name,
                    embeddings=self.embeddings
                    )
        return qdrant_obj

    def tiktoken_len(self, text):
        """
        텍스트의 token 길이 측정

        Args:
            text (str): 토큰 길이를 측정할 텍스트
        
        Returns:
            int: 입력된 텍스트의 토큰 길이
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)

    def get_chunks(self, text):
        """
        RecursiveCharacterTextSplitter로 텍스트 분할

        Args:
            text (str): 분할된 텍스트

        Returns:
            list: 분할된 텍스트 청크 리스트
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=self.tiktoken_len,
            add_start_index=True
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def add_vectorstore(self, client, text_chunks, collection_name, ids, metadatas):
        """
        벡터스토어에 텍스트 청크 추가
        """
        vectordb = self.create_qdrant_obj(client, collection_name)
        vectordb.add_texts(
                            text_chunks,
                            ids=ids,
                            metadatas=metadatas
                          )

    def qdrant_similarity_search(self, client, task, collection_name, filters=None, k=1):
        """
        task와 유사한 벡터 검색

        Args:
            client (QdrantClient): QdrantClient 연결 객체
            task (str): 유사도 검색할 작업
            collection_name (str): 벡터 컬렉션 이름
            filters (Optional[qdrant_client.http.models.Filter]): 필터링 조건, 기본값=None
            k (int): 반환할 유사한 벡터의 개수, 기본값=1

        Returns:
            List[qdrant_client.http.models.ScoredResult]: 유사한 벡터 검색 결과
        """
        vectordb = self.create_qdrant_obj(client, collection_name)
        search_result = vectordb.similarity_search_with_score(
            query=task,
            k=k,
            filter=filters)
        return search_result

def get_filter(key, value):
    """
    주어진 key와 value로 필터 생성

    Args:
        key (str): 필터링할 필드의 키
        value: 필터링할 값

    Returns:
        qdrant_client.http.models.Filter: 생성된 필터 객체
    """
    filters = models.Filter(
        must = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            )
        ]
    )
    return filters

def format_docs(docs):
    """
    주어진 문서 목록을 지정된 형식으로 포맷팅

    Args:
        docs (list): 문서 객체 목록
    
    Returns:
        str: 포맷팅된 문서
    """
    return " ".join([doc.page_content for doc in docs])

def remove_overlaps_in_sequence(texts):
    """
    주어진 텍스트 리스트에서 겹치는 부분을 제거한 후 합친 텍스트를 반환

    Args:
        texts (list): 겹치는 부분을 제거하고 합칠 텍스트 리스트
    
    Returns:
        str: 겹치는 부분이 제거된 후 합쳐진 텍스트
    """
    if not texts:
        return ""

    merged_text = texts[0]

    for next_text in texts[1:]:
        min_length = min(len(merged_text), len(next_text))
        for i in range(min_length, 0, -1):
            # merged_text와 next_text의 겹치는 부분을 제외하고 연결
            if merged_text.endswith(next_text[:i]):
                merged_text += next_text[i:]
                break
        else:
            merged_text += next_text

    return merged_text
