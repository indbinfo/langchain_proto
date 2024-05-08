"""
    - 변경사항 
        - 독스트링 추가
        - 메서드 인자들 개행 : 가독성 증가
        - 함수명 변경 : Python의 함수 명명 규칙에 맞추어 snake case 방식으로 교체
        - 실행 가능 코드 추가 : 모듈의 기능을 독립적으로 테스트할 수 있고, 
            다른 개발자들이 코드의 사용 방법을 더 쉽게 이해할 수 있음
"""

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ModelLoader:
    def __init__(self, model_id, model_path, task):
        """
        지정된 모델을 사용하여 ModelLoader를 초기화합니다.

        매개변수:
        model_id (str): 로드할 모델의 식별자입니다.
        model_path (str): 모델 캐시가 저장될 디렉터리 경로입니다.
        task (str): 모델이 최적화될 특정 작업입니다.
        """
        self.model_id = model_id
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=model_path, device_map='auto'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=model_path, device_map='auto'
        )

    # loadModel -> load_model 함수명 변경
    def load_model(self, prompt, max_new_tokens, repetition_penalty, top_k, do_sample=False):
        """
        특정 추론 설정으로 모델을 로드하고 파이프라인 연동 및 체인 반환

        매개변수:
        prompt (str): 모델에 입력될 텍스트입니다.
        max_new_tokens (int): 생성할 최대 토큰 수입니다.
        repetition_penalty (float): 텍스트 반복에 대한 패널티입니다.
        top_k (int): top-k 필터링을 위해 유지될 최고 확률의 어휘 토큰 수입니다.
        do_sample (bool): 샘플링을 사용할지 여부입니다. 기본값은 False입니다.

        반환값:
        Chain: 모델의 출력을 처리하는 체인입니다.
        """
        pipe = pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            top_k=top_k,
            device_map='auto'
        )

        huggingface_pipeline = HuggingFacePipeline(pipeline=pipe)
        output_chain = prompt | huggingface_pipeline | StrOutputParser()

        return output_chain
    
# ModelLoader의 사용 예
# 이 코드 모듈이 직접 실행될 때만 작동
if __name__ == '__main__':
    model_loader = ModelLoader("gpt3.5", "path", "text-generation")
    result_chain = model_loader.load_model("Hello, world!", 100, 1.2, 10, True)
    output = result_chain.execute()
    print(output)