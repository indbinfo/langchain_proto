from langchain.prompts import PromptTemplate
from multiprocessing import Process
import torch
# import 시 파일 전체가 아닌 필요 함수를 지정해서 가져오기
from gpu_performance import MetricsCollector
from load_model import ModelLoader

class ModelInference:
    def __init__(self, task, model_id, model_path, prompt_path, max_new_tokens, question):
        self.task = task
        self.model_id = model_id
        self.model_path = model_path
        self.prompt_path = prompt_path
        self.gpu_monitor_process = None
        self.max_new_tokens = max_new_tokens
        self.question = question

    # 모델 로드
    # 외부에서 사용하지 않는 method는 _로 시작하게 하여 구분
    def _load_model(self):
        self.modelLoader = ModelLoader(
            task=self.task,
            model_id=self.model_id,
            model_path=self.model_path,
        # 들여쓰기는 함수 시작한 라인에 맞추기
        )
        
    # 프롬프트 로드
    # 외부에서 사용하지 않는 method는 _로 시작하게 하여 구분
    def _load_prompt(self):
        with open(self.prompt_path) as f:
            template_text = f.read()
        self.prompt_template = PromptTemplate.from_template(template_text)
    # method 안에 함수를 넣지 않기
    def _monitor_gpu_performances(model_id, output_file, interval_sec):
        collector = MetricsCollector()
        # argument간 간격은 한칸씩 띄어주기
        collector.record_gpu_performance(model_id, output_file, interval_sec)
    # GPU 사용량 측정(multiprocess)
    def _start_gpu_monitoring(self, output_file, interval_sec=10):
        self.gpu_monitor_process = Process(target=self._monitor_gpu_performance, args=(self.model_id, output_file, interval_sec))
        self.gpu_monitor_process.start()

    def _stop_gpu_monitoring(self):
        if self.gpu_monitor_process:
            self.gpu_monitor_process.terminate()
            self.gpu_monitor_process.join()
            self.gpu_monitor_process = None
        
    # 모델 추론 시간 측정 및 결과 도출
    # method의 결과값과 과정에 맞춰서 이름 짓기(동사와 목적어 위주)
    def _run_pipeline_and_check_time(self):
        self._load_prompt()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # 부등호와 변수 사이 띄어주기
        result_txt = self.modelLoader.create_chain(
            prompt = self.prompt_template,
            max_new_tokens = self.max_new_tokens,
            do_sample = True, # 다음에 올 토큰에 대한 확률분포에 따라 단어 샘플링하여 문장 완성
            repetition_penalty = 1.1, # 중복된 결과값 통제(>1)
            top_k = 1,
        ).invoke({'question' : self.question})

        end_event.record()
        torch.cuda.synchronize()
        # 성능 측정에 사용된 시간은 latency 사용
        latency = round(start_event.elapsed_time(end_event) / 1000, 3)
        print(f'Inference Time: {latency} sec')

        return latency, result_txt
    
    # 실행
    def run(self, gpu_output_file, result_output_file):
        self._start_gpu_monitoring(gpu_output_file)
        self._load_model()
        latency, result_txt = self._run_pipeline_and_check_time()
        self._stop_gpu_monitoring()

        with open(result_output_file, 'w', encoding='utf-8') as f:
            # 포맷팅 규격은 f-string 방법 사용
            f.write('{self.model_id} {inference_time} sec\n{result}')

        print(result.split('end')[0])

# main.py는 모듈의 앞단에 호출 로직이 필요한 경우 쓰는 것으로 변경
if __name__ == '__main__':

    import time
    import json
    import os

    with open(config_path, 'r') as f:
        config = json.load(f)
    root_dir = config['path']['root_dir']
    # 경로 설정은 os 명령어 사용
    config_path = os.path.join(root_dir,'/config/config.json')
    # list, tuple 등의 iterable 객체를 활용하는 경우 명사 뒤에 s를 붙이기
    models_configs = [
        {
            'task':'text-generation',
            'model_id' : 'OpenBuddy/openbuddy-mistral-7b-v13',
            'model_path' : config['model_path'],
            'prompt_path' : '/opt/sample_prompt/prompt1/input/eng-kr.txt',
            'max_new_tokens' : 200,
            'question' : 'Translate the following text'
        },
    ]

    for config in models_configs:
        # 파일명 형식 지정
        datetime = time.strftime('%Y%m%d%H%M%S', time.localtime())
        gpu_output_file = os.path.join(root_dir, f'/log/gpu_{datetime}.log')
        result_output_file = os.path.join(root_dir, f'/result/result_{datetime}.txt')
        inference_model = ModelInference(**config)
        # 실행 결과 확인
        inference_model.run(
            gpu_output_file=gpu_output_file, # gpu 성능 결과 저장
            result_output_file=result_output_file # 모델 결과 저장
        )