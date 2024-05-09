import model_inference
import time
import json

if __name__ == '__main__':
    root_dir = '/home/llm/main/llm/'
    config_path = root_dir + 'config/config.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

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
        
        inference_model = model_inference.ModelInference(**config)

        # chain 반환
        # chain = inference_model.return_chain()
        
        # 실행 결과 확인
        inference_model.run(
            gpu_output_file=gpu_output_file, # gpu 성능 결과 저장
            result_output_file=result_output_file # 모델 결과 저장
        )