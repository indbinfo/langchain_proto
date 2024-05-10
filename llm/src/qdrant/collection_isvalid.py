"""
    - 변경사항
        - 독스트링 추가
        - config 파일 로드 예외 처리 추가 : 오류를 보다 쉽게 이해하고 명확한 에러 메세지 제공
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import qdrant_client
import numpy as np
import json

# 설정 파일 로드
config_path = '/home/llm/langchain_proto/web_main/config/qdrant.json'

try:
    with open(config_path, 'r', encoding='utf8') as f:
        config = json.load(f)
        QDRANT_HOST = config['QDRANT_HOST']
        QDRANT_API_KEY = config['QDRANT_API_KEY']
         # 설정값 검증
        if not QDRANT_HOST or not QDRANT_API_KEY:
            raise ValueError("QDRANT_HOST 또는 QDRANT_API_KEY가 설정 파일에 없습니다.")

        client = QdrantClient(
            QDRANT_HOST, 
            api_key=QDRANT_API_KEY
        )
except FileNotFoundError: # config 파일이 존재하지 않을 때
    print(f"설정 파일을 찾을 수 없습니다: {config_path}")
except Exception as e: # 그 외 모든 예외 처리
    print(f"설정 파일을 불러오는데 실패하였습니다.: {e}")

class VectorDB:
    def __init__(self, client, collection_name, size=768):
        """
        벡터 데이터베이스 관리를 위한 클래스를 초기화함

        매개변수:
        client (QdrantClient): Qdrant 클라이언트 객체.
        collection_name (str): 컬렉션 이름.
        size (int): 벡터의 차원 수.
        """
        self.client = client
        self.collection_name = collection_name
        self.size = size
    
    def get_embedding(self):
        """임베딩 모델을 로드합니다."""
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings

    def create_collection(self):
        """Qdrant에 새로운 컬렉션을 생성합니다."""
        collection_config = qdrant_client.http.models.VectorParams(
            size=self.size,
            distance=qdrant_client.http.models.Distance.COSINE
        )

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=collection_config
        )

    def add_vectorstore(self, text_chunks, ids, metadatas):
        """텍스트 청크를 벡터 스토어에 추가합니다."""
        vectorstore = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.get_embedding()
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.get_embedding()
        )
        vectorstore.add_texts(
            text_chunks,
            ids=ids,
            metadatas=metadatas
        )
        return vectorstore


if __name__ == "__main__":
    collection_name = 'question'
    vectordb = VectorDB(collection_name=collection_name, size=768, client=client)

    vectordb.create_collection()

    questions_file_path = '/home/llm/langchain_proto/vec/questions.json'

    # 유·무효질문
    try:
        with open(questions_file_path, 'r', encoding='utf8') as f:
            questions = json.load(f)
    except FileNotFoundError: # 질문 파일이 존재하지 않을 때
        print(f"파일을 찾을 수 없습니다: {questions_file_path}")
    except Exception as e: # 그 외 에러
        print(f"질문 파일을 불러오는 데 실패했습니다: {e}")

    # '무효질문' 카테고리의 질문 수만큼 딕셔너리가 반복되어 배열 형태로 생성됨
    meta_1=list(np.repeat({"filter":"무효질문"},len(questions['무효질문'])))
    meta_2=list(np.repeat({"filter":"유효질문"},len(questions['유효질문'])))

    question_list = questions['무효질문'] + questions['유효질문']
    ids = list(range(len(meta_1) + len(meta_2)))

    # vectorstore 저장 -> 질문 리스트 전체를 저장
    vec = vectordb.add_vectorstore(
        text_chunks=question_list,
        ids=ids,
        metadatas=sum([meta_1, meta_2], [])
        )

    # payload 설정 -> 특정 데이터 포인트에 메타데이터를 추가
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
