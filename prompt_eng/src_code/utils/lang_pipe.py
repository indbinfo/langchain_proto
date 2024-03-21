import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import time

class LangChainTest:
	# prompt_task에 따라 어떤 모델 설정할지 config에 구성(모델 다 받은 후)
	# prompt_task에 따라 
	def __init__(self, mdl_task, model):
		self.model = model
		self.mdl_task = mdl_task


	def apply_chain(self,input_txt):
		prompt_format = "question:{question} {input_txt}"
		template = PromptTemplate.from_template(prompt_format)
		output_parser = StrOutputParser()
		chain = template | self.model | output_parser
		start_time = time.time()
		output = chain.invoke({"question":self.mdl_task,
								"input_txt":input_txt})
		latency = time.time() - start_time
		print(f'inference 시간:{latency}')
		output = re.sub('python\n','',output)
		output = re.sub('```','',output).strip()
		print(f'결과:{output}')
		return output
	
	def df_invoke(self, df):
		# value_ls = []
		# for row in df['input']:
		# 	val = self.apply_chain(row)
		# 	val.append(value_ls)
		# df['output'] = val
		df['output'] = df['input'].apply(self.apply_chain)

		return df
	

		

