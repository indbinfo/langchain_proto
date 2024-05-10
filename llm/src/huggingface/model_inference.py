"""
변경사항
    - 독스트링 추가
    - 예외 처리 생성 
        - 로버스트성 증가 : 예상치 못한 상황에서 프로그램이 갑자기 중단없이 문제를 알려주거나 대응 가능
        - 사용자 경험 개선 : 오류 메시지를 통해 문제의 원인 파악 후 개선 가능
        - 디버깅 용이 : 오류의 원인을 좀 더 빨리 찾아 해결할 수 있음
    - 인자 간소화
        - ModelInference 클래스의 속성으로 존재하는 인자를 활용
        - 클래스 내부의 다른 메서드에서 이미 이 속성에 접근할 수 있기 때문에, 함수를 조금 더 간결하게 유지
        - 함수 내에서 'self.model_id'를 사용시, 이 함수가 클래스의 인스턴스 상태에 의존하고 있음이 명확해짐
    - 함수명 변경 : Python의 함수 명명 규칙에 맞추어 snake case 방식으로 교체
"""

from langchain.prompts import PromptTemplate
from multiprocessing import Process
import torch

import gpu_performance
import load_model

class ModelInference:
    def __init__(self, task, model_id, model_path, prompt_path, max_new_tokens, question):
        """
        모델 인퍼런스를 위한 클래스 초기화.

        Parameters:
        task (str): 모델이 수행할 태스크
        model_id (str): 모델의 고유 ID
        model_path (str): 모델 파일 경로
        prompt_path (str): 프롬프트 파일 경로
        max_new_tokens (int): 생성할 최대 토큰 수
        question (str): 사용자 질문
        """
        self.task = task
        self.model_id = model_id
        self.model_path = model_path
        self.prompt_path = prompt_path
        self.gpu_monitor_process = None
        self.max_new_tokens = max_new_tokens
        self.question = question

    def load_model(self):
        """
        모델 로딩을 담당하는 메소드.
        """
        self.model_loader = load_model.ModelLoader(
            task=self.task,
            model_id=self.model_id,
            model_path=self.model_path,
        )

    # 에러 처리 강화
    def load_prompt(self):
        """
        프롬프트 파일을 로드하여 템플릿 생성.
        """
        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as file:
                template_text = file.read()
            self.prompt_template = PromptTemplate.from_template(template_text)
        except FileNotFoundError: # 파일이 존재하지 않을 때
            print(f"Error: The file {self.prompt_path} does not exist.")
        except Exception as e: # 그 외 모든 예외 처리
            print(f"An error occurred: {e}")

    def start_gpu_monitoring(self, output_file, interval_sec=10):
        """
        GPU 모니터링을 멀티 프로세스로 시작.(GPU 사용량 측정)

        Parameters:
        output_file (str): 로그 파일 경로
        interval_sec (int): 로깅 간격(초)
        """
        def monitor_gpu_performance(output_file, interval_sec): # 기존에 있던 model_id 인자 제거함
            gpuPerformance = gpu_performance.GpuPerformance()
            gpuPerformance.record_gpu_performance(self.model_id, output_file, interval_sec)

        # Process를 사용하면 메인 프로그램의 실행흐름과 독립적으로 GPU 모니터링을 별도의 작업으로 수행 가능
        self.gpu_monitor_process = Process(target=monitor_gpu_performance, args=(output_file, interval_sec))
        self.gpu_monitor_process.start()

    def stop_gpu_monitoring(self):
        """
        GPU 모니터링 종료.
        """
        if self.gpu_monitor_process:
            self.gpu_monitor_process.terminate()
            self.gpu_monitor_process.join()
            self.gpu_monitor_process = None

    def perform_inference(self):
        """
        모델을 사용하여 추론을 수행하고, 추론 시간을 측정.

        Returns:
        tuple: (추론 시간, 추론 결과)
        """
        self.load_prompt()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        result = self.model_loader.load_model(
            prompt=self.prompt_template,
            max_new_tokens=self.max_new_tokens,
            do_sample=True, # 다음에 올 토큰에 대한 확률분포에 따라 단어 샘플링하여 문장 완성
            repetition_penalty=1.1, # 중복된 결과값 통제(>1)
            top_k=1,
        ).invoke({'question': self.question})
        ).invoke({'question': self.question})

        end_event.record()
        torch.cuda.synchronize()
        inference_time = round(start_event.elapsed_time(end_event) / 1000, 3)
        print(f'Inference Time: {inference_time} sec')

        return inference_time, result

    def run(self, gpu_output_file, result_output_file):
        """
        인퍼런스 프로세스 전체를 실행.

        Parameters:
        gpu_output_file (str): GPU 사용량 로그 파일 경로
        result_output_file (str): 결과가 저장될 파일 경로
        """
        self.start_gpu_monitoring(gpu_output_file)
        self.load_model()
        inference_time, result = self.perform_inference()
        self.stop_gpu_monitoring()

        with open(result_output_file, 'w', encoding='utf-8') as file:
            file.write(f'{self.model_id} {inference_time} sec\n{result}')

        print(result.split('end')[0])


