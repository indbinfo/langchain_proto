import os, json
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

# 테스트 주석
key_path = "/home/prompt_eng/main/prompt_eng/src_code/config/key.json"
with open(key_path,'r') as file:
	key_config = json.load(file)

config_path = "/home/prompt_eng/main/prompt_eng/src_code/config/config.json"
with open(config_path,'r') as file:
	config = json.load(file)

openai_key = key_config['key']['OPENAI_API_KEY']

class LangChainTest:
	# prompt_task에 따라 어떤 모델 설정할지 config에 구성(모델 다 받은 후)
	# prompt_task에 따라 
	def __init__(self, llm, pt_task, max_tokens, mdl_task,model=None, key=None):
		self.llm = llm
		self.pt_task = pt_task
		self.model = model
		self.max_tokens = max_tokens
		self.key = key
		self.mdl_task = mdl_task
		
		if self.llm == 'openai':
			self.key = openai_key
			self.model = ChatOpenAI(openai_api_key = openai_key,
								model = config['model']['openai'],
								temperature = 0
								)

	def apply_chain(self,input_txt):
		prompt_format = input_txt + "{question}"
		template = PromptTemplate.from_template(prompt_format)
		output_parser = StrOutputParser()
		chain = template | self.model | output_parser
		output = chain.invoke({"question":self.mdl_task})
		output = re.sub('python\n','',output)
		output = re.sub('```','',output).strip()

		return output

	def df_invoke(self, df):
		df['output'] = df['input'].apply(self.apply_chain)

		return df
	

		

