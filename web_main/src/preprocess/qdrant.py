import json
from qdrant_client import QdrantClient
import qdrant_client
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import tiktoken

with open(
    "/home/prompt_eng/langchain/langchain_proto/web_main/config/qdrant.json",
    "r",
    encoding="utf8",
) as f:
    config = json.load(f)

QDRANT_HOST = config["QDRANT_HOST"]
QDRANT_API_KEY = config["QDRANT_API_KEY"]


class VectorDB:
    def __init__(self):
        self.host = QDRANT_HOST
        self.api_key = QDRANT_API_KEY
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_connection(self):
        """
        QdrantClient와의 연결을 생성합니다.

        Returns:
            QdrantClient: 호스트와 API 키를 사용하여 생성된 QdrantClient의 인스턴스입니다.

        Note:
            이 함수는 QdrantClient와의 연결을 설정하는 데 사용됩니다. QdrantClient는 호스트와 API 키를 기반으로 데이터를 관리하고 처리합니다.
        """
        client = QdrantClient(self.host, api_key=self.api_key)
        return client

    def create_collection(self, qd_client, collection_name):
        """
        QdrantClient를 사용하여 새로운 컬렉션을 생성합니다.

        Args:
            qd_client (QdrantClient): QdrantClient의 인스턴스
            collection_name (str): 생성할 컬렉션의 이름

        Note:
            이 함수는 QdrantClient를 사용하여 새로운 컬렉션을 생성합니다. 새로운 컬렉션의 구성은 기본값으로 설정되며,
            크기는 768이며, 거리 메트릭은 코사인 거리로 설정됩니다.

        Warning:
            이 함수는 기존에 같은 이름의 컬렉션이 있으면 해당 컬렉션을 덮어쓰게 됩니다.
            따라서 컬렉션을 생성하기 전에 기존 컬렉션이 있는지 확인해야 합니다.
        """
        collection_config = qdrant_client.http.models.VectorParams(
            size=768,  # 1536 for OpenAI #768 for Gemini/HuggingFace/instructor-xl
            distance=qdrant_client.http.models.Distance.COSINE,
        )
        qd_client.recreate_collection(
            collection_name=collection_name, vectors_config=collection_config
        )

    def create_qdrant_obj(self, client, collection_name):
        """
        Qdrant 객체를 생성합니다.

        Args:
            client (QdrantClient): Qdrant와 통신하기 위한 클라이언트 객체
            collection_name (str): Qdrant에 저장될 컬렉션의 이름

        Returns:
            Qdrant: Qdrant 객체로, 클라이언트와 컬렉션 이름, 그리고 임베딩 데이터를 가지고 있습니다.

        Note:
            이 함수는 주어진 클라이언트와 컬렉션 이름을 사용하여 새로운 Qdrant 객체를 생성합니다.
            생성된 Qdrant 객체는 클라이언트와 통신할 때 사용되며, 지정된 컬렉션에 임베딩 데이터를 저장하거나 쿼리를 실행합니다.
        """
        qdrant_obj = Qdrant(
            client=client, collection_name=collection_name, embeddings=self.embeddings
        )
        return qdrant_obj

    def tiktoken_len(self, text):
        """
        주어진 텍스트의 토큰 길이를 반환합니다.

        Args:
            text (str): 토큰 길이를 계산할 텍스트

        Returns:
            int: 텍스트의 토큰 길이

        Note:
            이 함수는 주어진 텍스트를 처리하여 토큰 길이를 반환합니다. 토크나이저는 'cl100k_base'를 사용합니다.
            토큰 길이는 텍스트를 인코딩할 때 생성되는 토큰의 개수를 나타냅니다.
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)

    def get_chunks(self, text):
        """
        주어진 텍스트를 청크로 분할하여 반환합니다.

        Args:
            text (str): 분할할 텍스트

        Returns:
            list: 텍스트를 분할한 청크들의 리스트

        Note:
            이 함수는 주어진 텍스트를 주어진 설정에 따라 청크로 분할합니다.
            청크 크기는 400이며, 청크 간의 중첩은 50입니다.
            청크를 분할하는 데 사용되는 길이 함수는 'self.tiktoken_len'으로 지정되어 있습니다.
            각 청크에는 시작 인덱스가 추가됩니다.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=self.tiktoken_len,
            add_start_index=True,
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def add_vectorstore(self, client, text_chunks, collection_name, ids, metadatas):
        """
        텍스트 청크를 벡터 저장소에 추가합니다.

        Args:
            client (QdrantClient): 벡터 저장소와 통신하는 데 사용되는 클라이언트 객체
            text_chunks (list): 텍스트 청크들의 리스트
            collection_name (str): 텍스트를 저장할 컬렉션의 이름
            ids (list): 텍스트에 대한 고유 식별자(ID)들의 리스트
            metadatas (list): 텍스트에 대한 메타데이터들의 리스트

        Note:
            이 함수는 주어진 클라이언트를 사용하여 텍스트 청크들을 지정된 컬렉션에 추가합니다.
            각 텍스트 청크는 해당하는 고유 식별자(ID)와 메타데이터와 함께 저장됩니다.
        """
        # metadatas = list(np.repeat(payload_dict,len(text_chunks)))
        vectordb = self.create_qdrant_obj(client, collection_name)
        vectordb.add_texts(text_chunks, ids=ids, metadatas=metadatas)
        # client.set_payload(
        #                     collection_name=collection_name,
        #                     payload=payload_dict,
        #                     points=ids
        #                   )

    # def add_vectorstore(self, client, text_chunks, collection_name,ids, metadatas,payload_dict):
    # Similarity search with score by filter
    def qdrant_similarity_search(self, client, task, collection_name, filter=None, k=1):
        """
        Qdrant에서 유사도 검색을 수행하고 결과를 반환합니다.

        Args:
            client (QdrantClient): Qdrant와 통신하는 데 사용되는 클라이언트 객체입니다.
            task (str): 유사도를 검색할 쿼리나 작업입니다.
            collection_name (str): 검색을 수행할 컬렉션의 이름입니다.
            filter (str, optional): 검색 결과를 필터링하는 데 사용할 필터링 조건입니다. 기본값은 None입니다.
            k (int, optional): 검색 결과로 반환할 최대 항목 수입니다. 기본값은 1입니다.

        Returns:
            dict: 유사도 검색 결과를 나타내는 사전입니다. 키는 검색된 벡터의 ID이고, 값은 해당 벡터와의 유사도입니다.

        Note:
            이 함수는 주어진 클라이언트와 컬렉션 이름을 사용하여 Qdrant에서 유사도 검색을 수행합니다.
            검색 결과는 검색된 벡터의 ID와 해당 벡터와의 유사도를 포함하는 사전 형태로 반환됩니다.
        """
        vectordb = self.create_qdrant_obj(client, collection_name)
        search_result = vectordb.similarity_search_with_score(
            query=task, k=k, filter=filter
        )
        return search_result


def get_filter(key, value):
    """
    주어진 키와 값으로 필터를 생성하여 반환합니다.

    Args:
        key (str): 필터링할 필드의 키입니다.
        value (str): 필터링할 값입니다.

    Returns:
        Filter: 주어진 키와 값으로 생성된 필터 객체입니다.

    Note:
        이 함수는 주어진 키와 값으로 필터를 생성합니다. 생성된 필터는 주어진 키와 값에 해당하는 필드의 조건을 나타냅니다.
    """
    qdrant_filter = models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )
    return qdrant_filter


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
