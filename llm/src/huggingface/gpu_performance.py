"""
변경사항 
    - 독스트링 추가
    - 불필요한 'f.close()' 제거 :  with문 사용시 파일은 자동으로 닫힘
"""

import subprocess
import time
import re

class GpuPerformance:
    def record_gpu_performance(self, model_id, output_file, interval_sec):
        """
        정해진 간격으로 GPU 사용량을 기록하고 파일에 저장합니다.

        매개변수:
        model_id (str): 모델 식별자.
        output_file (str): GPU 사용량 기록이 저장되는 파일 경로.
        interval_sec (int): 기록 간의 시간 간격(초).
        """
        try:
            with open(output_file, 'w') as f:
                while True:
                    gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
                    pattern = r"\|\s+(\d)\s+Tesla V100-PCIE-32GB.*?\|\s+(\d+MiB\s+/\s+\d+MiB)\s+\|"
                    matches = re.findall(pattern, gpu_info, re.DOTALL) # 추출
                    memory_usage = [f"GPU {match[0]}: {match[1].replace(' ', '')}" for match in matches]
                    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    f.write(f"[{now}]: {model_id} {memory_usage}\n")
                    f.flush()
                    time.sleep(interval_sec)
                    
        finally:
            print('file closed.')
            # f.close() 코드 제거