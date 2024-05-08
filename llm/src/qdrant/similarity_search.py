import json
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http import models


# 설정 파일 로드 및 클라이언트 초기화
config_path = '/home/llm/main/llm/config/qdrant.json'

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

class QAResponse:
    def __init__(self, model_id, collection_name, client, task):
        """
        질문 응답 처리를 위한 클래스 초기화.

        매개변수:
        model_id (str): 모델 식별자.
        collection_name (str): 컬렉션 이름.
        client (QdrantClient): Qdrant 클라이언트 객체.
        task (str): 처리할 작업 또는 질문.
        """
        self.model_id = model_id
        self.collection_name = collection_name
        self.client = client
        self.task = task

    def count(self):
        """
        컬렉션 내 항목의 수를 카운트합니다.
        """
        client = self.client
        cnt = client.count(
            collection_name = self.collection_name,
            exact = True,
        )
        return cnt
    
    def get_embedding(self):
        """
        임베딩 모델을 로드합니다.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    
    def load_llm(self):
        """
        지정된 모델을 로드합니다.
        """
        return Ollama(model=self.model_id)
    
    def get_filter(self, key, value):
        """
        검색 필터를 생성합니다.

        매개변수:
        key (str): 필터 키.
        value (str): 매치될 값.
        """
        filter = models.Filter(
        must=[models.FieldCondition(
            key=key, 
            match = models.MatchValue(value=value))
            ]
        )
        return filter
    
    def qdrant_qa_response(self):
        """
        질문에 대한 응답을 검색 및 생성합니다.
        """
        vectordb = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.get_embedding()
        )
        llm = self.load_llm()

        # 검색강화
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
        )

        response = qa.invoke({'query' : self.task})
        return response
    
    def is_valid(self, result):
        """
        결과의 유효성을 검사합니다.

        매개변수:
        result (list): 검색 결과 리스트.
        """
        valid_yn = True
        for data in result:
            valid_filter = data[0].metadata['filter']
            if valid_filter == "무효질문":
                valid_yn = False
                break
        if valid_yn == False:
            return "요청 주신 질문에 대해서 답변이 어렵습니다."
        else:
            return "답변 제공"
    
    def qdrant_similarity_search(self, k, score_threshold, filter):
        """
        유사도 검색을 수행합니다.

        매개변수:
        k (int): 반환될 결과의 수.
        score_threshold (float): 점수 임계값.
        filter (Filter): 적용할 필터.
        """
        vectordb = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.get_embedding()
            )
        
        search_result = vectordb.similarity_search_with_score(
            query=self.task, 
            k=k,
            score_threshold =score_threshold, 
            filter=filter
            )
        
        return search_result

config = {
    'model_id': 'wizardcoder:34b-python',
    'collection_name': 'context_ys',
    'client': client,
    'task': sys.argv[1]
    }

qa_response = QAResponse(**config)

# 첫 번째 유사도 검색 실행
result_k1 = qa_response.qdrant_similarity_search(
    k=1, 
    score_threshold=0,
    filter=None
)

document, score = result_k1[0]
filter_value = document.metadata['filter']
print(filter_value)

# 두 번째 유사도 검색을 실행하여, 특정 필터 값에 대한 결과를 100개 가져옴
search_result = qa_response.qdrant_similarity_search(
    score_threshold=0,
    k=100,
    filter=qa_response.get_filter(
        key='filter', 
        value=filter_value
        ),
    )

# 결과를 _id 메타데이터를 기준으로 정렬
sorted_result = sorted(search_result, key=lambda x: x[0].metadata['_id'])

# 정렬된 결과 출력
for item in sorted_result:
    document = item[0]
    print(document.metadata['_id'], document.metadata['filter'])