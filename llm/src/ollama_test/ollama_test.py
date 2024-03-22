from langchain_community.llms import Ollama
import ollama
import torch
import json
import time

root_dir = '/home/llm/main/llm/'

def config(root_dir):
    with open(root_dir + 'config/config.json') as f:
        config = json.load(f)

    model_path = config['path']['model_path']
    prompt_path = config['path']['prompt_path'] + 'prompt1/input/'

    return model_path, prompt_path

def load_file(path, filename):
    with open(path + filename) as f:
        file = f.read()
    return file

def load_model_list():
    model_list = [i['model'] for i in ollama.list()['models']]
    return model_list

def inference(prompt, model, temperature):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    llm = Ollama(model=model, temperature=temperature)
    result = llm.invoke(prompt)

    end_event.record()
    torch.cuda.synchronize()

    inference_time = round(start_event.elapsed_time(end_event) / 1000, 3)
    return result, inference_time

model_path, prompt_path = config(root_dir=root_dir)
code = load_file(path=prompt_path, filename='code.txt')
kr_eng = load_file(path=prompt_path, filename='kr-eng.txt')
eng_kr = load_file(path=prompt_path, filename='eng-kr.txt')

model_id = 'mistral'

result, inference_time = inference(code, model_id, 0)
print(result, inference_time)

datetime = time.strftime('%Y%m%d%H%M%S', time.localtime())

output_file = root_dir + 'log/ollama_log.log'

with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n\n[{0} {1}] : {2} sec {3}'.format(datetime, model_id, inference_time, result))
