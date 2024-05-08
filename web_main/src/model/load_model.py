"""
* Version: 1.0
* 파일명: load_model.py
* 설명: 여러 플랫폼의 모델을 로드하는 모듈 (Ollama, HuggingFacePipeline, ChatOpenAI)
* 수정일자: 2024/05/02
* 수정자: 손예선
* 수정 내용
    1. 4개씩 indentation 되어있지 않은 경우 수정, Docstring 추가
    2. LocalModelLoader -> HuggingFaceModelLoader로 이름 변경
    3. 환경에 따라 경로 설정
"""

import os
import json
from langchain_openai import ChatOpenAI
from custom_ollama import Ollama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

if os.name == 'posix':
    PATH = os.path.join(
        os.environ.get('HOME'),
        'langchain_proto',
        'web_main',
    )
elif os.name == 'nt':
    PATH = os.path.join(
       'c:', 
       os.environ.get('HOMEPATH'),
       "langchain_proto", 
       "web_main",
    )
else:
    PATH = None

with open(os.path.join(PATH, 'config', 'key.json'),'r', encoding='utf-8') as file:
    key_config = json.load(file)

with open(os.path.join(PATH, 'config', 'config.json'),'r', encoding='utf-8') as file:
    config = json.load(file)


class OllamaModelLoader:
    """
    Ollama 모델 로드
    """
    def __init__(self, model_id, pt_task):
        self.model_id = model_id
        if pt_task == "eng-kr":
            self.pt_task = "kr-eng"
        else:
            self.pt_task = pt_task
        self.temperature = config["llama_model"][self.pt_task][model_id]['temperature']
        self.repeat_penalty = config["llama_model"][self.pt_task][model_id]['repeat_penalty']

    def load_model(self):
        """
        Ollama 모델을 로드하는 함수
        Return: 
            Ollama: Ollama 모델 객체
        """
        model = Ollama(model=self.model_id,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            )
        return model


class HuggingFaceModelLoader:
    """
    HuggingFace 모델 로드
    """
    def __init__(self, model_id, pt_task):
        self.task = "text-generation"
        self.model_id = model_id
        self.model_path = config['model'][pt_task]['model_path']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir=self.model_path)

    def load_model(self, max_new_tokens, repetition_penalty, top_k, do_sample=False):
        """
        HuggingFace 모델을 로드하는 함수

        Returns: 
            HuggingFacePipeline: HuggingFacePipeline 모델 객체
        """
        pipe = pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            top_k=top_k,
            device = 0,
            # device_map=auto
            )
        model = HuggingFacePipeline(pipeline=pipe)
        return model


class OpenaiModelLoader:
    """
    OpenAI의 ChatGPT 모델 로드
    """
    def __init__(self):
        self.key = key_config['key']['OPENAI_API_KEY']

    def load_model(self, temperature):
        """
        ChatOpenAI 모델을 로드하는 함수

        Returns:
            ChatOpenAI: ChatOpenAI 모델 객체
        """
        model = ChatOpenAI(openai_api_key = self.key,
                           model = config['model']['kr-eng']['model_id'],
                           temperature = temperature,
                        )
        return model
