import json
from langchain_openai import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from custom_ollama import Ollama

KEY_PATH = "/home/prompt_eng/langchain/langchain_proto/web_main/config/key.json"
with open(KEY_PATH, "r", encoding="utf-8") as file:
    key_config = json.load(file)

CONFIG_PATH = "/home/prompt_eng/langchain/langchain_proto/web_main/config/config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)


class OllamaModelLoader:
    def __init__(self, model_id, pt_task):
        self.model_id = model_id
        if pt_task == "eng-kr":
            self.pt_task = "kr-eng"
        else:
            self.pt_task = pt_task
        self.temperature = config["llama_model"][self.pt_task][model_id]["temperature"]
        self.repeat_penalty = config["llama_model"][self.pt_task][model_id][
            "repeat_penalty"
        ]

    def load_model(self):
        model = Ollama(
            model=self.model_id,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            # keep_alive=1
        )
        return model


class LocalModelLoader:
    def __init__(self, model_id, pt_task):
        self.task = "text-generation"
        self.model_id = model_id
        self.model_path = config["model"][pt_task]["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.model_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, cache_dir=self.model_path
        )

    def load_model(self, max_new_tokens, repetition_penalty, top_k, do_sample=False):
        pipe = pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            top_k=top_k,
            # device_map='auto'
            device=0,
        )
        model = HuggingFacePipeline(pipeline=pipe)

        return model


class OpenaiModelLoader:
    def __init__(self):
        self.key = key_config["key"]["OPENAI_API_KEY"]

    def load_model(self, temperature):
        model = ChatOpenAI(
            openai_api_key=self.key,
            model=config["model"]["kr-eng"]["model_id"],
            temperature=temperature,
        )
        return model
