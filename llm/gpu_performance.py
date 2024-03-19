import os
import sys
import re
import time
import subprocess

home_dir=os.environ['HOME']
date = time.strftime('%Y%m%d', time.localtime())
model_id = sys.argv[1]

def record_gpu_performance(output, interval_sec):
    try:
        with open(output, 'a') as f:
            while True:
                gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
                pattern = r"\|\s+(\d)\s+Tesla V100-PCIE-32GB.*?\|\s+(\d+MiB\s+/\s+\d+MiB)\s+\|"
                matches = re.findall(pattern, gpu_info, re.DOTALL)
                memory_usage = [f"GPU {match[0]}: {match[1].replace(' ', '')}" for match in matches]
                now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                f.write(f"[{now}]: {model_id} {memory_usage}\n")
                time.sleep(interval_sec)
                
    except KeyboardInterrupt:
        print('Recording Stopped.')

if __name__ == "__main__":
    output_file = home_dir+'/main/llm/log/gpu_performance_{0}.log'.format(date)
    interval_sec = 30
    record_gpu_performance(output_file, interval_sec)
