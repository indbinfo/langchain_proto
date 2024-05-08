from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class ModelLoader:
    def __init__(self, model_id, model_path, task):
        self.model_id = model_id
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path, device_map='auto')
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path, device_map='auto')
    # method명은 snake case를 사용하고 함수 목적에 더 가깝게 명명한다.
    def create_chain(self, prompt, max_new_tokens, repetition_penalty, top_k, do_sample=False):
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
        hf = HuggingFacePipeline(pipeline=pipe)
        chain = prompt | hf | StrOutputParser()

        return chain
    
