import subprocess
import time
import re


class GpuPerformance:
    def record_gpu_performance(self, model_id, output, interval_sec):
        try:
            with open(output, 'w') as f:
                while True:
                    gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
                    pattern = r"\|\s+(\d)\s+Tesla V100-PCIE-32GB.*?\|\s+(\d+MiB\s+/\s+\d+MiB)\s+\|"
                    matches = re.findall(pattern, gpu_info, re.DOTALL)
                    memory_usage = [f"GPU {match[0]}: {match[1].replace(' ', '')}" for match in matches]
                    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    f.write(f"[{now}]: {model_id} {memory_usage}\n")
                    f.flush()
                    time.sleep(interval_sec)

        finally:
            print('file closed.')
            f.close()
