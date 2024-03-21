import json
from langchain_openai import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

key_path = "/home/prompt_eng/main/prompt_eng/src_code/config/key.json"
with open(key_path,'r') as file:
	key_config = json.load(file)

config_path = "/home/prompt_eng/main/prompt_eng/src_code/config/config.json"
with open(config_path,'r') as file:
	config = json.load(file)


class LocalModelLoader:
    def __init__(self, task):
        self.task = task
        self.model_id = config['model'][task]['model_id']
        self.model_path = config['model'][task]['model_path']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir=self.model_path)
        
    
    def load_model(self, max_new_tokens, repetition_penalty, top_k, do_sample=False):
        pipe = pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            top_k=top_k,
            device_map='auto'
        )    
        model = HuggingFacePipeline(pipeline=pipe)

        return model


class OpenaiModelLoader:
    def __init__(self):
        self.key = key_config['key']['OPENAI_API_KEY']
    def load_model(self, temperature):      
        model = ChatOpenAI(openai_api_key = self.key,
                            model = config['model']['kr-eng']['model_id'],
                            temperature = temperature,
                            )        
        return model