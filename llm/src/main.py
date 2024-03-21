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
            'task':'en2ko',
            'model_id' : 'traintogpb/llama-2-en2ko-translator-7b-qlora-bf16-upscaled',
            'model_path' : config['model_path'],
            'prompt_path' : '/opt/sample_prompt/prompt1/input/eng-kr.txt'
        },

        {
            'task' : 'codeGen',
            'model_id' : 'codellama/CodeLlama-7b-Python-hf',
            'model_path' : config['model_path'],
            'prompt_path' : '/opt/sample_prompt/prompt1/input/code.txt'
        }
    ]

    for config in models_config:
        # 파일명 형식 지정
        datetime = time.strftime('%Y%m%d%H%M%S', time.localtime())
        gpu_output_file = root_dir + f'log/gpu_{datetime}.log'
        result_output_file = root_dir + f'result/result_{datetime}.txt'
        
        # 실행
        inference_model = model_inference.ModelInference(**config)
        inference_model.run(
            gpu_output_file=gpu_output_file, # gpu 성능 결과 저장
            result_output_file=result_output_file # 모델 결과 저장
        )