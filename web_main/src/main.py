import json
from preprocess.load_data import load_template
from model.load_model import OllamaModelLoader
config_path = "/home/prompt_eng/langchain/langchain_proto/web_main/config/config.json"

with open (config_path) as f:
    config = json.load(f)

# template 로드
kr_eng_input = ['context','task']
code_gen_input = ['context','task']
eng_kr_input = ['context','task']



kr_eng_temp = load_template(template_file= config['template_file']['kr-eng'], input_variables=kr_eng_input)
code_gen_temp = load_template(config['template_file']['code-gen'], input_variables=code_gen_input)
eng_kr_temp = load_template(config['template_file']['eng-kr'], input_variables=eng_kr_input)

# 모델 로드
kr_eng_loader = OllamaModelLoader(model_id = "mistral",
                        pt_task = "kr-eng")
kr_eng_mdl = kr_eng_loader.load_model()

code_gen_loader = OllamaModelLoader(model_id = "wizardcoder:34b-python",
                        pt_task = "code-gen")
code_gen_mdl = kr_eng_loader.load_model()

eng_kr_loader = OllamaModelLoader(model_id = "mistral",
                        pt_task = "eng-kr")
eng_kr_mdl = eng_kr_loader.load_model()


