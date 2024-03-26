from langchain.prompts import PromptTemplate
from multiprocessing import Process
import torch

import gpu_performance
import load_model

class ModelInference:
    def __init__(self, task, model_id, model_path, prompt_path):
        self.task = task
        self.model_id = model_id
        self.model_path = model_path
        self.prompt_path = prompt_path
        self.gpu_monitor_process = None

    # 모델 로드
    def load_model(self):
        self.modelLoader = load_model.ModelLoader(
            model_id=self.model_id,
            model_path=self.model_path,
            )
        
    # 프롬프트 로드
    def load_prompt(self):
        with open(self.prompt_path) as f:
            template_text = f.read()
        self.prompt_template = PromptTemplate.from_template(template_text)

    # GPU 사용량 측정(multiprocess)
    def start_gpu_monitoring(self, output_file, interval_sec=10):
        def monitor_gpu_performance(model_id, output_file, interval_sec):
            gpuPerformance = gpu_performance.GpuPerformance()
            gpuPerformance.record_gpu_performance(model_id, output_file, interval_sec)

        self.gpu_monitor_process = Process(target=monitor_gpu_performance, args=(self.model_id, output_file, interval_sec))
        self.gpu_monitor_process.start()

    def stop_gpu_monitoring(self):
        if self.gpu_monitor_process:
            self.gpu_monitor_process.terminate()
            self.gpu_monitor_process.join()
            self.gpu_monitor_process = None
        
    # 모델 추론 시간 측정 및 결과 도출
    def perform_inference(self):
        self.load_prompt()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if self.task == 'en2ko':
            question = f'Translate the following text.'
        elif self.task == 'codeGen':
            question = f'부산의 동래구 중 많은 매출을 올린 업종을 막대그래프로 그려줘.'
        
        result = self.modelLoader.loadModel(
            prompt=self.prompt_template,
            max_new_tokens=200,
            do_sample=True,
            repetition_penalty=1.1, # 중복된 결과값 통제(>1)
            top_k=1,
        ).invoke({'question' : question})

        end_event.record()
        torch.cuda.synchronize()
        inference_time = round(start_event.elapsed_time(end_event) / 1000, 3)
        print(f'Inference Time: {inference_time} sec')

        return inference_time, result
    
    # 실행
    def run(self, gpu_output_file, result_output_file):
        self.start_gpu_monitoring(gpu_output_file)
        self.load_model()
        inference_time, result = self.perform_inference()
        self.stop_gpu_monitoring()

        with open(result_output_file, 'w', encoding='utf-8') as f:
            f.write('{0} {1} sec :\n{2}'.format(self.model_id, inference_time, result.split('end')[0]))

        print(result.split('end')[0])