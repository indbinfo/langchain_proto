from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import sys

with open('/home/llm/main/llm/config/qdrant.json', 'r', encoding='utf8') as f:
    config = json.load(f)

QDRANT_HOST = config['QDRANT_HOST']
QDRANT_API_KEY = config['QDRANT_API_KEY']

client = QdrantClient(
    QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

class QAResponse:
    def __init__(self, model_id, collection_name, client, task):
        self.model_id = model_id
        self.collection_name = collection_name
        self.client = client
        self.task = task

    def count(self):
        client = self.client
        cnt = client.count(
            collection_name = self.collection_name,
            exact = True,
        )
        return cnt
    
    def get_embedding(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    
    def load_llm(self):
        return Ollama(model=self.model_id)
    
    def get_filter(self, key, value):
        filter = models.Filter(
        must=[models.FieldCondition(
            key=key, 
            match = models.MatchValue(value=value))
            ]
        )
        return filter
    
    def qdrant_qa_response(self):
        vectordb = Qdrant(
            client = self.client,
            collection_name = self.collection_name,
            embeddings = self.get_embedding()
        )
        llm = self.load_llm()
        qa = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type='stuff',
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
        )

        response = qa.invoke({'query' : self.task})
        return response
    
    def isValid(self, result):
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
    'model_id' : 'wizardcoder:34b-python',
    'collection_name' : 'context_ys',
    'client' : client,
    'task' : sys.argv[1]
    }

qa_response = QAResponse(**config)

result_k1 = qa_response.qdrant_similarity_search(
    k=1, 
    score_threshold=0,
    filter=None
)

document, score = result_k1[0]
filter_value = document.metadata['filter']
print(filter_value)

search_result = qa_response.qdrant_similarity_search(
    score_threshold=0,
    k=100,
    filter=qa_response.get_filter(
        key='filter', 
        value=filter_value
        ),
    )

sorted_result = sorted(search_result, key=lambda x: x[0].metadata['_id'])

for item in sorted_result:
    document = item[0]
    print(document.metadata['_id'], document.metadata['filter'])