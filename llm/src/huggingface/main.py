"""
변경사항 
    - 독스트링 추가 : 각 함수에 해당 기능을 설명하는 독스트링 추가
    - 모듈화 및 함수화 : 설정 파일을 로드하는 기능 함수로 분리하여 코드의 모듈성을 높였고,
        전체 실행 흐름을 main 함수 내에 정의하여 if __name__ == '__main__' 블록이 간결해지도록 함
"""

import model_inference
import time
import json

def load_config(path):
    """
    주어진 경로의 JSON 설정 파일을 로드합니다.
    
    Parameters:
    path (str): 설정 파일의 경로
    
    Returns:
    dict: 설정 파일에서 로드된 설정 정보
    """
    with open(path, 'r') as file:
        return json.load(file)


def main():
    """
    주어진 모델 설정에 따라 인퍼런스를 수행하고, 결과 및 GPU 사용량을 로그 파일로 저장합니다.
    모든 설정은 외부 JSON 파일에서 로드됩니다.
    """
    root_dir = '/home/llm/main/llm/'
    config_path = root_dir + 'config/config.json'
    config = load_config(config_path)
    
    models_config = [
        {
            'task':'text-generation',
            # 'model_id' : 'traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled',
            # 'model_id' : 'Salesforce/codegen25-7b-mono',
            # 'model_id' : 'Salesforce/codegen2-7B',
            'model_id' : 'OpenBuddy/openbuddy-mistral-7b-v13',
            'model_path' : config['model_path'],
            'prompt_path' : '/opt/sample_prompt/prompt1/input/eng-kr.txt',
            'max_new_tokens' : 200,
            'question' : 'Translate the following text'
        },
        
        # {
        #     'task' : 'text-generation',
        #     'model_id' : 'TheBloke/CodeLlama-7B-Python-GPTQ',
        #     'model_path' : config['model_path'],
        #     'prompt_path' : '/opt/sample_prompt/prompt1/input/code.txt',
        #     'max_new_tokens' : 200,
        #     'question' : '부산의 동래구 중 많은 매출을 올린 업종을 막대그래프로 그려줘.'
        # },
        
    ]
    
    for config in models_config:
        # 파일명 형식 지정
        datetime = time.strftime('%Y%m%d%H%M%S', time.localtime())
        gpu_output_file = root_dir + f'log/gpu_{datetime}.log'
        result_output_file = root_dir + f'result/result_{datetime}.txt'
        
        # 모델 인퍼런스 인스턴스 생성 및 실행
        inference_model = model_inference.ModelInference(**config)

        # chain 반환
        # chain = inference_model.return_chain()
        
        # 실행 결과 확인
        inference_model.run(
            gpu_output_file=gpu_output_file, # gpu 성능 결과 저장
            result_output_file=result_output_file # 모델 결과 저장
        )
    
    
if __name__ == '__main__':
    main()